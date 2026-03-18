"""Deck Nine AdventureTools shared color palette.

Charcoal backgrounds with warm amber/gold accent — shared across
bodypipe, ShootDay, Pebble, StoryForge, and ShootDayPost.
"""

COLORS = {
    # Backgrounds — dark-on-dark layering
    "bg_primary":     "#31363b",  # Main window, panels, chrome
    "bg_input":       "#232629",  # Line edits, text areas, tables, trees
    "bg_scroll":      "#2A2929",  # Scrollbar tracks
    "bg_active_tab":  "#54575B",  # Selected tab panel
    # Text
    "text_primary":   "#eff0f1",  # Primary text everywhere
    "text_secondary": "#76797C",  # Subtle labels
    "text_disabled":  "#454545",  # Disabled text
    # Accent — gold/amber signature
    "accent":         "#ca952e",  # Focus, hover borders, selected tabs, progress
    "accent_pressed": "#ab7e28",  # Button pressed, selection bg
    "accent_active":  "#ba8826",  # Active item selection
    "hover":          "#685126",  # Item hover
    # Borders & chrome
    "border":         "#76797C",  # Frame borders, separators
    "border_disabled":"#454545",  # Disabled borders
    # Slider / scrollbar
    "slider_grip":    "#605F5F",  # Handle elements
    "slider_groove":  "#565a5e",  # Slider tracks
    # Special surfaces
    "tooltip_bg":     "#5A7566",  # Tooltip background
    "header_checked": "#334e5e",  # Sorted/checked column headers
    "toolbar_ext":    "#58595a",  # Toolbar overflow
    # Semantic status (unchanged — readable on both backgrounds)
    "success":        "#4ecca3",
    "warning":        "#ffd93d",
    "error":          "#ff6b6b",
}
