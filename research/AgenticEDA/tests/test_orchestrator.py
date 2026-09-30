"""Smoke tests for the multi-loop orchestrator architecture."""
import inspect
import unittest


class TestRunAgentSignature(unittest.TestCase):
    def test_accepts_new_params(self):
        from src.agent.eda_agent import run_agent
        sig = inspect.signature(run_agent)
        assert "max_subagents" in sig.parameters
        assert "max_loops" in sig.parameters
        assert "loop_timeout" in sig.parameters
        assert sig.parameters["max_subagents"].default == 3
        assert sig.parameters["max_loops"].default == 2
        assert sig.parameters["loop_timeout"].default == 180


class TestAgentStateFields(unittest.TestCase):
    def test_has_loop_fields(self):
        from src.agent.state import AgentState
        state = AgentState(dataset_path="/tmp/test.csv", session_id="test")
        assert hasattr(state, "loop_count")
        assert hasattr(state, "subagent_run_count")
        assert hasattr(state, "max_subagents")
        assert hasattr(state, "max_loops")
        assert state.loop_count == 0
        assert state.max_subagents == 3


class TestSubagentSignature(unittest.TestCase):
    def test_accepts_kernel_id(self):
        from src.agent.subagent import run_subagent
        sig = inspect.signature(run_subagent)
        assert "kernel_id" in sig.parameters
        assert "notebook_id" in sig.parameters

    def test_investigation_result_has_images(self):
        from src.agent.subagent import InvestigationResult
        result = InvestigationResult(hypothesis_id="h1", hypothesis_title="T", finding="F")
        assert hasattr(result, "images")
        assert isinstance(result.images, dict)
        assert result.status == "complete"
        assert result.to_dict()["status"] == "complete"


class TestKGIntegration(unittest.TestCase):
    def test_kg_used_in_eda_agent(self):
        with open("src/agent/eda_agent.py") as f:
            source = f.read()
        assert "KnowledgeGraph" in source
        assert "get_context_for_hypothesis_generation" in source
        assert "find_similar_hypothesis" in source
        assert "KernelPoolManager" in source
        assert "loop_start" in source
        assert "subagent_start" in source

    def test_hypothesis_accepts_kg_context(self):
        from src.agent.hypothesis import generate_hypotheses
        sig = inspect.signature(generate_hypotheses)
        assert "kg_context" in sig.parameters


class TestKernelPool(unittest.TestCase):
    def test_pool_importable(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        assert hasattr(pool, "allocate_subagent_kernels")
        assert hasattr(pool, "shutdown_subagent_kernels")
        assert hasattr(pool, "inject_dataset_preamble")


class TestRunEndpointConfig(unittest.TestCase):
    def test_accepts_config_body(self):
        with open("backend/routers/run.py") as f:
            source = f.read()
        assert "RunConfig" in source
        assert "max_subagents" in source
        assert "max_loops" in source


class TestSteeringIntegration(unittest.TestCase):
    def test_run_config_accepts_research_direction(self):
        from backend.routers.run import RunConfig

        config = RunConfig(research_direction="focus on churn")

        assert config.research_direction == "focus on churn"

    def test_run_agent_accepts_research_direction(self):
        from src.agent.eda_agent import run_agent

        sig = inspect.signature(run_agent)

        assert "research_direction" in sig.parameters
        assert sig.parameters["research_direction"].default is None


class TestSteeringAgentState(unittest.TestCase):
    def test_agent_state_has_steering_fields(self):
        from src.agent.state import AgentState

        state = AgentState(
            dataset_path="/tmp/test.csv",
            session_id="test",
            research_direction="focus on downtime",
        )
        state.steering_notes.append({"content": "compare weekend behavior"})

        assert state.research_direction == "focus on downtime"
        assert isinstance(state.steering_notes, list)
        assert isinstance(state.consumed_steering_ids, list)
        assert "focus on downtime" in state.run_guidance()
        assert "compare weekend behavior" in state.run_guidance()

    def test_reasoning_accepts_guidance(self):
        from src.agent.reasoning import decide_next_step

        sig = inspect.signature(decide_next_step)

        assert "run_guidance" in sig.parameters

    def test_hypothesis_generation_accepts_guidance(self):
        from src.agent.hypothesis import generate_hypotheses

        sig = inspect.signature(generate_hypotheses)

        assert "run_guidance" in sig.parameters

    def test_subagent_accepts_guidance(self):
        from src.agent.subagent import run_subagent

        sig = inspect.signature(run_subagent)

        assert "run_guidance" in sig.parameters


class TestChatKGIntegration(unittest.TestCase):
    def test_chat_has_kg_support(self):
        with open("backend/routers/chat.py") as f:
            source = f.read()
        assert "set_session_kg" in source
        assert "_session_kgs" in source
        assert "find_similar_hypothesis" in source
        assert "subagent_complete" in source

    def test_chat_context_accepts_kg(self):
        from src.chat.chat_agent import ChatContext
        sig = inspect.signature(ChatContext.__init__)
        assert "kg" in sig.parameters


class TestVisionLoop(unittest.TestCase):
    def test_interpret_output_uses_content_parts(self):
        with open("src/agent/reasoning.py") as f:
            source = f.read()
        # The fix: HumanMessage(content=content_parts) instead of plain string
        assert "HumanMessage(content=content_parts)" in source


if __name__ == "__main__":
    unittest.main()
