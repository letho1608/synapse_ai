"""
Behavioral tests for the redesigned TUI look (display helpers + a render).

Checks what users see without a terminal: structured transcript lines
(cards, replies, banner) keep their exact text while gaining classes,
every glyph renders in both Windows console fonts, the bottom bar fits the
width, menus highlight the selection, and a headless render of the full
app draws the rounded boxes, the placeholder and the status bar.
"""
import time

import pytest

from synapse import console as C
from synapse import theme as TH
from synapse import tui as T

# Characters present in BOTH Cascadia Mono and Consolas (checked with
# fontTools on the Windows fonts); anything else draws as a tofu box.
SAFE_EXTRA = set("›▸●○×│╭╮╰╯─·•…")


def _ctx():
    c = C.ConsoleCtx(port=52415, defaults_file=None, node_label="nodeA",
                     model_short="SmolLM2", modes="int8",
                     model_id="HuggingFaceTB/SmolLM2-135M-Instruct")
    for i in range(3):
        c.logbuf.append("[node] line %d\n" % i)
    return c


@pytest.fixture(autouse=True)
def _fake_bar(monkeypatch):
    monkeypatch.setattr(C, "fetch_bar_data", lambda port: {"node": "nodeA", "peers": 2})
    T._bar_cache.update(ts=0.0)


def _texts(frags):
    return "".join(t for _, t in frags)


@pytest.mark.parametrize("line", [
    T.card_header("/status", "14:02:11"),
    T.CARD_INDENT + "[ok] cluster healthy",
    T.CARD_INDENT + "[fail] boom",
    "  " + T.G_GUTTER,
    T._footer(0, 1.0), T._footer(2, 1.0),
    T.G_OK + " reply text", "  continued reply",
    T.G_ECHO + " hello", "> legacy echo", "[node] log", "[console] boot", "plain",
] + T.banner_lines("0.1.0", "m", 52415, "synapse").split("\n"))
def test_style_line_never_changes_text(line):
    frags = T.style_line(line)
    assert _texts(frags) == line  # cursor/selection math relies on this


def test_structured_lines_get_their_classes():
    head = T.style_line(T.card_header("/status", "14:02:11"))
    assert head[0] == ("class:card.head", T.G_CARD + " /status")
    assert head[1][0] == "class:card.time"
    body = T.style_line(T.CARD_INDENT + "[fail] boom")
    assert body[0][0] == "class:gutter" and body[1][0] == "class:err"
    assert T.style_line(T._footer(0, 1))[0][0] == "class:card.ok"
    assert T.style_line(T._footer(1, 1))[0][0] == "class:card.err"
    assert T.style_line(T.G_OK + " hi")[0][0] == "class:reply.dot"


def test_card_and_reply_formatting():
    assert T.card_body("a\n\nb") == "\n".join(
        [T.CARD_INDENT + "a", "  " + T.G_GUTTER, T.CARD_INDENT + "b"])
    assert T.reply_body("x\ny") == T.G_OK + " x\n  y"
    assert T.reply_body("  \n") == ""


def test_banner_is_a_closed_box_of_equal_width():
    from prompt_toolkit.utils import get_cwidth
    lines = T.banner_lines("0.1.0", "HuggingFaceTB/SmolLM2-135M-Instruct", 52415, "synapse").split("\n")
    widths = {sum(get_cwidth(ch) for ch in ln) for ln in lines}
    assert widths == {64}
    assert lines[0].startswith("╭") and lines[-1].startswith("╰")


def test_every_ui_glyph_is_font_safe():
    produced = "".join([
        T.card_header("/x", "00:00:00"), T.card_body("a"), T._footer(0, 1), T._footer(1, 1),
        T.reply_body("a"), T.banner_lines("v", "m", 1, "t"), T.G_ECHO, "".join(T.SPINNER)])
    ctx = _ctx()
    st = T.TuiState()
    st.job = {"label": "/benchmark", "t0": time.monotonic()}
    produced += _texts(T.panel_fragments(ctx, st)) + _texts(T.status_fragments(ctx, st, 140))
    produced += _texts(T.menu_fragments([("/a", "b")] * 12, 0)) + _texts(T.help_fragments())
    odd = {ch for ch in produced if ord(ch) > 127 and ch not in SAFE_EXTRA}
    assert not odd, odd


@pytest.mark.parametrize("cols", [60, 100, 160])
def test_status_bar_fills_exactly_the_width(cols):
    from prompt_toolkit.utils import get_cwidth
    frags = T.status_fragments(_ctx(), T.TuiState(), cols)
    width = sum(get_cwidth(t) for _, t in frags)
    assert width == cols or (width > cols and not any("quit" in t for _, t in frags))


def test_status_bar_reflects_state():
    ctx, st = _ctx(), T.TuiState()
    assert "online" in _texts(T.status_fragments(ctx, st, 120))
    st.job = {"label": "/benchmark", "t0": time.monotonic() - 5}
    assert "running 5s" in _texts(T.status_fragments(ctx, st, 120))


def test_panel_has_sections_and_trims_log_prefix():
    ctx, st = _ctx(), T.TuiState()
    text = _texts(T.panel_fragments(ctx, st))
    for sec in ("CLUSTER", "JOBS", "LOGS", "nodeA", "idle", "line 2"):
        assert sec in text
    assert "[node]" not in text
    ctx.logs_on = False
    assert "/logs on" in _texts(T.panel_fragments(ctx, st))


def test_menu_highlights_only_the_selection():
    frags = T.menu_fragments([("/a", "x"), ("/b", "y"), ("/c", "z")], 1)
    sel = [t for c, t in frags if c == "class:mrow.sel.name"]
    assert len(sel) == 1 and sel[0].strip() == "/b"
    assert _texts(T.menu_fragments([], 0)).strip() == "(no match)"
    many = T.menu_fragments([("/%d" % i, "d") for i in range(15)], 0, max_rows=10)
    assert "5 more" in _texts(many)


def test_theme_defines_every_new_class():
    style = TH.build_style(TH.load_theme("synapse"))
    for cls in ("gutter", "card.head", "card.ok", "card.err", "reply.dot", "box.border",
                "box.border.active", "section", "placeholder", "mrow.sel", "mrow.sel.name",
                "status.logo", "status.ok", "status.busy", "status.hint"):
        assert style.get(cls), cls
    # never the prompt_toolkit built-in "menu" class (grey background)
    assert not any(k == "menu" or k.startswith("menu.") for k in style)


def test_headless_render_draws_boxes_placeholder_and_bar(tmp_path, monkeypatch):
    import asyncio
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setattr(T, "HISTORY_FILE", str(tmp_path / "hist"))

    class Out(DummyOutput):
        def get_size(self):
            return Size(rows=30, columns=120)

    async def run():
        with create_pipe_input() as inp:
            ctx = _ctx()
            ui = T._build_ui(type("A", (), {})(), ctx, None, None,
                             asyncio.get_running_loop(), app_input=inp, app_output=Out())
            ui.say(T._banner_text(None, ctx))
            ui.say(T.card_header("/status", "14:00:00"))
            ui.say(T.card_body("[ok] fine"))
            ui.say(T._footer(0, 0.2))
            shot = {}

            async def grab():
                await asyncio.sleep(0.3)
                screen = ui.app.renderer._last_screen
                shot["rows"] = ["".join(screen.data_buffer[y][x].char for x in range(120))
                                for y in range(30)]
                ui.app.exit(result=0)
            asyncio.ensure_future(grab())
            await ui.app.run_async()
            return shot["rows"]

    rows = asyncio.run(run())
    screen = "\n".join(rows)
    assert "╭" in screen and "╯" in screen
    assert "Ask the model, or type / for commands" in screen
    assert rows[-1].startswith(" synapse ") and "online" in rows[-1]
    assert "CLUSTER" in screen and "done" in screen
