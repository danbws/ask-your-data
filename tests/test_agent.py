import pytest

from ask_your_data.agent import Agent
from ask_your_data.providers import AnthropicProvider, FakeProvider, ProviderError, demo_provider
from ask_your_data.providers.fake import final_text, tool_call


def test_demo_script_end_to_end_with_recovery(db):
    result = Agent(demo_provider(), db).ask("which machine re-dyes the most?")

    assert result.stop_reason == "end_turn"
    assert [s.tool for s in result.steps] == ["run_sql", "run_sql", "run_python", "make_chart"]
    # first query is wrong on purpose; the agent loop must surface the error and carry on
    assert result.steps[0].is_error and "redo_flag" in result.steps[0].output
    assert not result.steps[1].is_error
    # the answer is computed from real tool output, and matches the database
    top = db.run_select(
        "SELECT machine_id FROM dye_batches GROUP BY machine_id "
        "ORDER BY 1.0 * SUM(redye) / COUNT(*) DESC LIMIT 1"
    ).rows[0][0]
    assert result.answer.startswith(top)
    assert result.chart_png and result.chart_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(result.sql) == 4


def test_tool_results_go_back_in_one_user_message_per_turn(db):
    seen = []

    def spy(messages):
        seen.append([m["role"] for m in messages])
        return final_text("done")

    provider = FakeProvider([
        tool_call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM machines"}, "a"),
        spy,
    ])
    result = Agent(provider, db).ask("how many machines?")
    assert seen == [["user", "assistant", "user"]]
    assert result.answer == "done" and "12" in result.steps[0].output


def test_max_steps_guard_stops_a_looping_model(db):
    looping = FakeProvider([
        tool_call("run_sql", {"sql": "SELECT 1"}, f"c{i}") for i in range(20)
    ])
    result = Agent(looping, db, max_steps=3).ask("loop forever")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3 and looping.calls == 3
    assert "Stopped after 3 steps" in result.answer


def test_guarded_write_attempt_becomes_error_result_not_a_write(db):
    provider = FakeProvider([
        tool_call("run_sql", {"sql": "DROP TABLE machines"}, "w"),
        final_text("I could not do that."),
    ])
    result = Agent(provider, db).ask("drop the machines table")
    assert result.steps[0].is_error
    assert db.run_select("SELECT COUNT(*) FROM machines").rows == [(12,)]


def test_unknown_tool_is_an_error_result(db):
    provider = FakeProvider([tool_call("rm_rf", {}, "x"), final_text("ok")])
    result = Agent(provider, db).ask("q")
    assert result.steps[0].is_error and "unknown tool" in result.steps[0].output


def test_system_prompt_contains_the_schema(db):
    captured = {}

    class Spy:
        name = "spy"

        def complete(self, system, messages, tools):
            captured["system"] = system
            captured["tools"] = [t["name"] for t in tools]
            return final_text("hi")

    Agent(Spy(), db).ask("hello")
    assert "TABLE dye_batches" in captured["system"]
    assert captured["tools"] == ["run_sql", "run_python", "make_chart"]


# --- AnthropicProvider with an injected stub client (no network) -----------------------------


class _Block:
    def __init__(self, data):
        self._data = data

    def model_dump(self, exclude_none=False):
        return self._data


class _Response:
    def __init__(self, blocks, stop_reason):
        self.content = [_Block(b) for b in blocks]
        self.stop_reason = stop_reason


class _StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_anthropic_provider_drives_the_loop_with_the_right_request_shape(db):
    client = _StubClient([
        _Response([{"type": "tool_use", "id": "tu_1", "name": "run_sql",
                    "input": {"sql": "SELECT COUNT(*) AS n FROM machines"}}], "tool_use"),
        _Response([{"type": "text", "text": "There are 12 machines."}], "end_turn"),
    ])
    provider = AnthropicProvider(client=client)
    result = Agent(provider, db).ask("how many machines?")

    assert result.answer == "There are 12 machines."
    first, second = client.requests
    assert first["model"] == "claude-sonnet-5-5"
    assert "tool_choice" not in first and "temperature" not in first
    assert [t["name"] for t in first["tools"]] == ["run_sql", "run_python", "make_chart"]
    # second request carries the assistant tool_use and the matching tool_result
    last_user = second["messages"][2]  # [user q, assistant tool_use, user tool_result]
    assert last_user["role"] == "user"
    assert last_user["content"][0]["tool_use_id"] == "tu_1"
    assert last_user["content"][0]["is_error"] is False


def test_anthropic_provider_maps_sdk_errors():
    import anthropic
    import httpx

    class Boom:
        messages = None

        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.AuthenticationError(
                "bad key", response=httpx.Response(401, request=req), body=None)

    with pytest.raises(ProviderError, match="API key"):
        AnthropicProvider(client=Boom()).complete("s", [{"role": "user", "content": "q"}], [])
