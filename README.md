# Floorplan Visualization Tool v2

A Python tool for creating multi-floor architectural plans from JSON definitions, with special support for visualizing vertical alignment across floors.

## Quick Start

```bash
python floorplan.py your_manor.json output.svg
```

## Features

- **Arbitrary polygon rooms** - L-shapes, hexagons, octagons, any shape
- **Vertical elements** - Elevators, chimneys, lightwells spanning multiple floors
- **Stair connections** - Explicit floor-to-floor connections
- **Alignment guides** - Visual aids for checking structural wall alignment
- **Auto-calculated areas** - Room areas computed via shoelace formula
- **Floor percentage tracking** - Shows what % of each floor vertical elements consume
- **Label customization** - Per-element font, size, color, and positioning

## JSON Format

### Basic Structure

```json
{
  "name": "Building Name",
  "label_defaults": {...},
  "floors": [...],
  "vertical_elements": [...],
  "alignment_guides": [...]
}
```

### Global Label Defaults

Set building-wide label styling:

```json
{
  "label_defaults": {
    "font": "Georgia, serif",
    "size": 11,
    "color": "#333333"
  }
}
```

### Rooms (Any Polygon Shape)

```json
{
  "name": "L-Shaped Kitchen",
  "vertices": [[0, 6], [6, 6], [6, 9], [8, 9], [8, 12], [0, 12]],
  "doors": [{"wall": 0, "position": 0.5, "width": 1.2}],
  "windows": [{"wall": 5, "position": 0.5, "width": 0.8}],
  "label": {
    "font": "Times New Roman, serif",
    "size": 14,
    "color": "#8e44ad",
    "show_area": true,
    "show_name": true,
    "offset_x": 0,
    "offset_y": -1.5
  }
}
```

**Label options:**
- `font` - CSS font-family string (e.g., `"Georgia, serif"`)
- `size` - Font size in pixels (absolute if ≥5, multiplier if <5)
- `color` - CSS color (e.g., `"#c0392b"` or `"darkred"`)
- `show_area` - Whether to display the area (default: true)
- `show_name` - Whether to display the name (default: true)
- `offset_x`, `offset_y` - Position adjustment in meters

Vertices define the polygon clockwise or counter-clockwise. Wall indices count from 0 (edge from vertex 0 to vertex 1).

**Non-rectangular examples:**
- L-shape: 6 vertices
- Hexagonal turret: 6 vertices  
- Octagonal bay window: 8 vertices
- Pentagonal tower: 5 vertices

### Stairs (with Floor Connections)

```json
{
  "vertices": [[17, 10], [21, 10], [21, 14], [17, 14]],
  "connects": [0, 1],
  "name": "Main Stairs",
  "label": {"size": 10}
}
```

The `connects` array specifies which two floors the staircase links. Stairs are drawn on both connected floors with appropriate up/down indicators.

### Vertical Elements

Define once, automatically rendered on all specified floors:

```json
{
  "type": "elevator",
  "name": "service elevator", 
  "vertices": [[10, 5.5], [12, 5.5], [12, 7.5], [10, 7.5]],
  "floors": [-1, 0, 1, 2],
  "hidden": true,
  "label": {"size": 9, "show_area": false}
}
```

**Supported types:**
- `elevator` - Purple, or red if `hidden: true`
- `chimney` - Brown
- `lightwell` - Cyan
- `shaft` - Gray
- `dumbwaiter` - Purple

The `hidden` flag renders the element with a distinct color (red) and dashed pattern, useful for secret or bricked-over features.

### Alignment Guides

Polylines to help verify structural walls align across floors:

```json
{
  "name": "East structural wall",
  "points": [[14, 0], [14, 14]],
  "floors": [-1, 0, 1, 2]
}
```

Guides render as dashed orange lines on specified floors.

## Output

The tool generates SVG with:
- 2×2 grid layout (configurable via `grid_columns`)
- Color-coded elements with legend
- Floor titles with total area
- Vertical elements showing area and floor range
- Alignment guides as dashed overlay

## Console Summary

Running the tool prints a detailed summary:

```
Building: Manor
==================================================

Level 0: Ground Floor (18th c. + 19th c.)
  Rooms: 7, Total area: 312.0 m²
    - Entry Hall: 30.0 m²
    - Library: 40.0 m²
    ...
  Stairs:
    - Main Stairs: L0 ↔ L1 (16.0 m²)

Vertical Elements:
  - Hidden Lift (elevator): 4.0 m² on floors -1, 0, 1, 2 [HIDDEN]
      L0: 1.28% of floor area
      L1: 1.30% of floor area
```

This helps verify your hidden shaft consumes the right percentage of floor space.

## Configuration

Edit the `Config` class in `floorplan.py`:

```python
@dataclass
class Config:
    scale: float = 30.0           # pixels per meter
    grid_columns: int = 2         # 2x2 layout; use 1 for vertical strip
    
    # Label defaults
    label_font: str = "Arial, sans-serif"
    label_font_size: float = 11.0
    label_color: str = ""         # empty = use element color
    
    # Colors
    wall_color: str = "#2d3436"
    door_color: str = "#d35400"
    window_color: str = "#3498db"
    stairs_color: str = "#27ae60"
    elevator_color: str = "#9b59b6"
    elevator_hidden_color: str = "#e74c3c"
    chimney_color: str = "#795548"
    # ... etc
```

## Label Styling Hierarchy

Labels are resolved in order of precedence:
1. **Element-level** `label` object (highest priority)
2. **Building-level** `label_defaults`
3. **Config class** defaults (lowest priority)

This allows you to set a consistent font across the building while overriding specific rooms.

## Workflow Tips

1. **Coordinate system**: Pick a corner as origin (0,0). All measurements in meters.

2. **Shared walls**: If two rooms share a wall, use identical vertex coordinates for that edge.

3. **Non-rectangular rooms**: Trace the perimeter, listing vertices in order. The shoelace formula handles any simple polygon.

4. **Checking the 2% rule**: The console output shows what percentage of each floor your hidden elements consume. Adjust dimensions or verify that adjacent "dead space" (thick walls, closets) accounts for discrepancies.

5. **Historical layers**: Use descriptive floor names. You can also use `fill_color` on rooms to visually distinguish 17th c. vs 19th c. construction.

6. **Alignment verification**: Add alignment guides along structural walls, then visually check the SVG to ensure walls line up across floors.

7. **Label positioning**: Use `offset_x` and `offset_y` to nudge labels away from walls or into better positions for odd-shaped rooms.

