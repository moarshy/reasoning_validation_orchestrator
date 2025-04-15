import logging
import os
import json
import asyncio
import traceback
import uuid
from dotenv import load_dotenv

from naptha_sdk.modules.agent import Agent
from naptha_sdk.schemas import OrchestratorRunInput, AgentRunInput, AgentDeployment
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from reasoning_validation_orchestrator.schemas import InputSchema

logger = logging.getLogger(__name__)

class ReasoningValidationOrchestrator:
    async def create(self, deployment, *args, **kwargs):
        logger.info(f"Creating orchestrator with deployment: {deployment}")
        
        if not hasattr(deployment, 'agent_deployments') or len(deployment.agent_deployments) < 2:
            logger.error(f"Missing or insufficient agent deployments. Found: {getattr(deployment, 'agent_deployments', None)}")
            try:
                config_path = "reasoning_validation_orchestrator/configs/deployment.json"
                logger.info(f"Attempting to load agent deployments from {config_path}")
                
                with open(config_path, "r") as f:
                    deployments_config = json.load(f)
                
                if isinstance(deployments_config, list) and len(deployments_config) > 0:
                    deployment_config = deployments_config[0]
                    
                    if "agent_deployments" in deployment_config and len(deployment_config["agent_deployments"]) >= 2:
                        # Create Agent deployment objects
                        agent_deployments = []
                        for agent_config in deployment_config["agent_deployments"][:2]:  # Take first two
                            agent_config["node"] = deployment.node.dict() if hasattr(deployment.node, 'dict') else deployment.node
                            agent_deployments.append(AgentDeployment(**agent_config))
                        
                        deployment.agent_deployments = agent_deployments
                        logger.info(f"Successfully loaded {len(agent_deployments)} agent deployments")
                    else:
                        raise ValueError("Configuration doesn't have enough agent deployments")
                else:
                    raise ValueError("Invalid deployment configuration format")
                
            except Exception as e:
                logger.error(f"Failed to load agent deployments from configuration: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError("Orchestrator requires exactly 2 agent deployments: reasoning and validation")
        
        if not hasattr(deployment, 'agent_deployments') or len(deployment.agent_deployments) < 2:
            raise ValueError("Orchestrator requires exactly 2 agent deployments: reasoning and validation")

        self.deployment = deployment
        self.agent_deployments = deployment.agent_deployments

        try:
            self.reasoning_agent = Agent()
            await self.reasoning_agent.create(deployment=self.agent_deployments[0], *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize reasoning agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[0]}")
            raise

        try:
            self.validation_agent = Agent()
            await self.validation_agent.create(deployment=self.agent_deployments[1], *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize validation agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[1]}")
            raise

    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        run_id = str(uuid.uuid4())
        private_key = get_private_key_from_pem(os.getenv("PRIVATE_KEY_FULL_PATH"))

        reasoning_input = {
            "func_name": "reason",
            "problem": module_run.inputs.problem,
            "num_thoughts": module_run.inputs.num_thoughts
        }

        reasoning_run_input = AgentRunInput(
            consumer_id=module_run.consumer_id,
            inputs=reasoning_input,
            deployment=self.agent_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, private_key)
        )

        try:
            reasoning_run = await self.reasoning_agent.run(reasoning_run_input)
            reasoning_result = reasoning_run.results[0] if reasoning_run.results else {}
            logger.info(f"Reasoning completed with {len(reasoning_result.get('thoughts', []))} thoughts")
        except Exception as e:
            logger.error(f"Error in reasoning step: {e}")
            logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise e

        validation_input = {
            "func_name": "validate",
            "problem": module_run.inputs.problem,
            "thoughts": reasoning_result['thoughts']
        }

        validation_run_input = AgentRunInput(
            consumer_id=module_run.consumer_id,
            inputs=validation_input,
            deployment=self.agent_deployments[1],
            signature=sign_consumer_id(module_run.consumer_id, private_key)
        )

        try:
            validation_result = await self.validation_agent.run(validation_run_input)
            validation_result = validation_result.results[0] if validation_result.results else {}
            best_thought_idx = validation_result.get('best_thought_index', 0)
            logger.info(f"Validation completed, selected best thought index: {best_thought_idx}")
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise e

        result = {
            "problem": module_run.inputs.problem,
            "reasoning_thoughts": reasoning_result['thoughts'],
            "validation_result": validation_result,
            "final_answer": validation_result.get('final_answer', '')
        }
        return result

async def run(module_run: dict, *args, **kwargs):
    module_run = OrchestratorRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    orchestrator = ReasoningValidationOrchestrator()
    await orchestrator.create(module_run.deployment, *args, **kwargs)
    result = await orchestrator.run(module_run, *args, **kwargs)
    return result

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "orchestrator",
        "reasoning_validation_orchestrator/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))

    input_params = {
        "problem": "What is the sum of all integers from 1 to 100?",
        "num_thoughts": 3
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
    }

    response = asyncio.run(run(module_run))
    print("Final Answer:", response.get("final_answer"))
    print("Best Reasoning Path:", response.get("validation_result", {}).get("best_thought", ""))