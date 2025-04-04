import logging
import os
import json
import asyncio
import traceback
import uuid
from dotenv import load_dotenv

from naptha_sdk.modules.agent import Agent
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.schemas import OrchestratorRunInput, AgentRunInput, KBRunInput
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from reasoning_validation_orchestrator.schemas import InputSchema

logger = logging.getLogger(__name__)

class ReasoningValidationOrchestrator:
    async def create(self, deployment, *args, **kwargs):
        # Handle deployment as dict or object (Pydantic model) with attributes
        if isinstance(deployment, dict):
            agent_deployments = deployment.get("agent_deployments")
            kb_deployments = deployment.get("kb_deployments")
        else:
            agent_deployments = deployment.agent_deployments
            kb_deployments = deployment.kb_deployments

        if not agent_deployments or len(agent_deployments) < 2:
            logger.error(f"Missing or insufficient agent deployments. Found: {agent_deployments}")
            raise ValueError("Orchestrator requires exactly 2 agent deployments: reasoning and validation")

        self.deployment = deployment
        self.agent_deployments = agent_deployments

        if not kb_deployments:
            logger.error(f"Missing KB deployments. Found: {kb_deployments}")
            raise ValueError("Orchestrator requires at least one KB deployment")
        self.kb_deployments = kb_deployments

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

        try:
            self.kb = KnowledgeBase()
            await self.kb.create(deployment=self.kb_deployments[0], *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize KB: {e}")
            logger.error(f"KB deployment details: {self.kb_deployments[0]}")
            raise

    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        run_id = str(uuid.uuid4())
        private_key = get_private_key_from_pem(os.getenv("PRIVATE_KEY_FULL_PATH"))

        init_result = await self.kb.run(KBRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "init"},
            deployment=self.kb_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, private_key)
        ))
        logger.info(f"KB Init result: {init_result}")

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
            reasoning_result = await self.reasoning_agent.run(reasoning_run_input)
            logger.info(f"Reasoning completed with {len(reasoning_result['thoughts'])} thoughts")
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
            best_thought_idx = validation_result.get('best_thought_index', 0)
            logger.info(f"Validation completed, selected best thought index: {best_thought_idx}")
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise e

        try:
            kb_data = {
                "problem": module_run.inputs.problem,
                "solution": validation_result.get('final_answer', ''),
                "reasoning": validation_result.get('best_thought', '')
            }
            add_data_result = await self.kb.run(KBRunInput(
                consumer_id=module_run.consumer_id,
                inputs={"func_name": "add_data", "func_input_data": kb_data},
                deployment=self.kb_deployments[0],
                signature=sign_consumer_id(module_run.consumer_id, private_key)
            ))
            logger.info(f"KB Add data result: {add_data_result}")
        except Exception as e:
            logger.error(f"Error storing results in KB: {e}")

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

    # Load deployment using the setup function.
    deployment = asyncio.run(setup_module_deployment(
        "orchestrator",
        "reasoning_validation_orchestrator/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))

    # If the deployment is missing agent or KB deployments, load and inject them from JSON.
    if not deployment.agent_deployments or not deployment.kb_deployments:
        with open("reasoning_validation_orchestrator/configs/deployment.json", "r") as f:
            deployments = json.load(f)
        if deployments and isinstance(deployments, list):
            deployment_dict = deployments[0]  # Choose the first deployment entry
            deployment.agent_deployments = deployment_dict.get("agent_deployments", [])
            deployment.kb_deployments = deployment_dict.get("kb_deployments", [])

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
