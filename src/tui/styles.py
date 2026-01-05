"""CSS theme for Loreguard TUI using Crush/Dracula colors."""

# Dracula color palette
PINK = "#FF79C6"
PURPLE = "#BD93F9"
CYAN = "#8BE9FD"
GREEN = "#50FA7B"
YELLOW = "#F1FA8C"
RED = "#FF5555"
ORANGE = "#FFB86C"
FG = "#F8F8F2"
FG_DIM = "#6272A4"
BG = "#282A36"
BG_DARK = "#1E1F29"
SELECTED_BG = "#44475A"

# CSS for the entire application
LOREGUARD_CSS = f"""
/* Only set background on non-modal screens */
MainScreen {{
    background: {BG};
}}

/* Banner styling */
LoreguardBanner {{
    width: 100%;
    height: auto;
    content-align: center middle;
    text-align: center;
    padding: 0;
}}

/* Main content area */
#main-content {{
    width: 100%;
    height: 1fr;
    padding: 1 2;
}}

#main-status {{
    color: {FG};
}}

/* Hardware info bar */
HardwareInfo {{
    width: 100%;
    height: 1;
    content-align: center middle;
    text-align: center;
    color: {FG_DIM};
    padding: 0 1;
}}

/* Status panel */
StatusPanel {{
    height: 100%;
    padding: 1;
}}

StatusPanel .status-line {{
    height: 1;
    color: {FG};
}}

StatusPanel .status-label {{
    color: {FG_DIM};
}}

StatusPanel .status-value {{
    color: {FG};
}}

StatusPanel RichLog {{
    height: 1fr;
    border: solid {FG_DIM};
    padding: 0 1;
}}

/* Modal dialogs */
.modal-dialog {{
    width: 50;
    max-width: 60;
    border: round {PURPLE};
    background: {BG};
    padding: 1 2;
}}

.modal-title {{
    text-align: center;
    text-style: bold;
    color: {PURPLE};
    padding-bottom: 1;
}}

.modal-footer {{
    text-align: center;
    color: {FG_DIM};
    padding-top: 1;
}}

/* Input styling */
Input {{
    background: {BG_DARK};
    border: solid {FG_DIM};
    padding: 0 1;
}}

Input:focus {{
    border: solid {PURPLE};
}}

/* ListView styling */
ListView {{
    background: transparent;
    padding: 0;
}}

ListView > ListItem {{
    padding: 0 1;
    height: 1;
}}

ListView > ListItem.-selected {{
    background: {PURPLE};
    color: {FG};
    text-style: bold;
}}

/* Footer */
Footer {{
    background: {BG_DARK};
    color: {FG_DIM};
    height: 1;
}}

Footer .footer--key {{
    color: {CYAN};
}}

/* Progress bars */
ProgressBar {{
    padding: 0 1;
}}

ProgressBar Bar {{
    color: {CYAN};
}}

ProgressBar PercentageStatus {{
    color: {FG_DIM};
}}

/* Buttons (minimal, keyboard-driven) */
Button {{
    background: {SELECTED_BG};
    color: {FG};
    border: none;
    padding: 0 2;
    min-width: 10;
}}

Button:focus {{
    background: {PURPLE};
    color: {FG};
}}

/* Command palette specific */
CommandPaletteModal {{
    align: center middle;
}}

CommandPaletteModal .command-item {{
    padding: 0 1;
}}

CommandPaletteModal .command-shortcut {{
    color: {FG_DIM};
    text-align: right;
}}

/* Token input modal */
TokenInputModal {{
    align: center middle;
}}

/* Model list for selection */
.model-item {{
    padding: 0 1;
}}

.model-name {{
    color: {FG};
}}

.model-size {{
    color: {FG_DIM};
}}

.model-warning {{
    color: {YELLOW};
}}

/* Auth screen */
AuthScreen {{
    align: center middle;
}}

/* Screen titles */
.screen-title {{
    text-align: center;
    text-style: bold;
    color: {PURPLE};
    padding: 1 0;
}}

.screen-subtitle {{
    text-align: center;
    color: {FG_DIM};
    padding-bottom: 1;
}}

/* Step indicator */
.step-indicator {{
    text-align: center;
    color: {FG_DIM};
    padding: 1 0;
}}

/* Error messages */
.error {{
    color: {RED};
    padding: 1;
}}

/* Success messages */
.success {{
    color: {GREEN};
    padding: 1;
}}

/* Warning messages */
.warning {{
    color: {YELLOW};
    padding: 1;
}}
"""
