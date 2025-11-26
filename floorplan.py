#!/usr/bin/env python3
"""
Floorplan visualization tool for multi-story buildings.
Generates SVG output from JSON room definitions.

Features:
- Multi-floor visualization with vertical alignment
- Vertical elements (elevators, chimneys) spanning multiple floors
- Stairs with explicit floor connections
- Alignment guides for checking structural consistency
- Auto-calculated room areas
- Arbitrary polygon room shapes

Version 2.0 - Enhanced vertical alignment support
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field


# =============================================================================
# Configuration & Color Palette
# =============================================================================

@dataclass
class Config:
    """Drawing configuration."""
    scale: float = 30.0  # pixels per meter
    
    # Element colors (4-color palette + extras)
    wall_color: str = "#2d3436"
    wall_width: float = 3.0
    door_color: str = "#d35400"
    door_width: float = 4.0
    window_color: str = "#3498db"
    window_width: float = 3.0
    stairs_color: str = "#27ae60"
    stairs_width: float = 2.0
    
    # Vertical element colors
    elevator_color: str = "#9b59b6"      # purple for elevators
    elevator_hidden_color: str = "#e74c3c"  # red for hidden/secret
    chimney_color: str = "#795548"       # brown for chimneys
    lightwell_color: str = "#00bcd4"     # cyan for lightwells
    shaft_color: str = "#607d8b"         # gray for generic shafts
    vertical_width: float = 2.5
    vertical_dash: str = "6,3"
    vertical_hidden_dash: str = "3,3"
    
    # Alignment guides
    guide_color: str = "#ff9800"
    guide_width: float = 1.0
    guide_dash: str = "10,5"
    
    # Labels and layout
    label_font: str = "Arial, sans-serif"
    label_font_size: float = 11.0
    label_color: str = ""  # empty = use element color
    floor_label_font_size: float = 16.0
    floor_padding: float = 2.0  # meters around each floor
    floor_spacing: float = 3.0  # meters between floors in layout
    background_color: str = "#fafafa"
    room_fill_opacity: float = 0.08
    grid_columns: int = 2  # 2x2 layout for 4 floors


@dataclass
class LabelStyle:
    """Label styling options."""
    font: Optional[str] = None
    size: Optional[float] = None  # absolute size, or multiplier if < 5
    color: Optional[str] = None
    show_area: bool = True  # for rooms: whether to show area
    show_name: bool = True  # whether to show name
    offset_x: float = 0.0  # offset in meters
    offset_y: float = 0.0


# =============================================================================
# Geometry Utilities
# =============================================================================

def shoelace_area(vertices: List[Tuple[float, float]]) -> float:
    """Calculate polygon area using the shoelace formula. Works for any polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the centroid of a polygon. Works for any polygon."""
    n = len(vertices)
    if n == 0:
        return (0, 0)
    if n == 1:
        return vertices[0]
    if n == 2:
        return ((vertices[0][0] + vertices[1][0]) / 2,
                (vertices[0][1] + vertices[1][1]) / 2)
    
    cx, cy = 0.0, 0.0
    signed_area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = vertices[i]
        x1, y1 = vertices[j]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    
    signed_area *= 0.5
    if abs(signed_area) < 1e-10:
        # Fallback to simple average for degenerate cases
        return (sum(v[0] for v in vertices) / n,
                sum(v[1] for v in vertices) / n)
    
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return (cx, cy)


def point_on_segment(p1: Tuple[float, float], p2: Tuple[float, float], 
                     t: float) -> Tuple[float, float]:
    """Get point at parameter t (0-1) along segment from p1 to p2."""
    return (p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]))


def segment_length(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def polygon_bounds(vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box of polygon: (min_x, min_y, max_x, max_y)."""
    if not vertices:
        return (0, 0, 0, 0)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Opening:
    """Door or window on a wall."""
    wall: int  # wall index (0 = first edge of polygon)
    position: float  # 0-1 along the wall
    width: float  # in meters
    type: str = "door"  # "door" or "window"


@dataclass 
class Stairs:
    """Staircase connecting two floors."""
    vertices: List[Tuple[float, float]]
    connects: Tuple[int, int]  # (lower_floor, upper_floor) indices
    name: str = ""
    bidirectional: bool = True  # drawn on both connected floors
    label: Optional[LabelStyle] = None
    
    @property
    def area(self) -> float:
        return shoelace_area(self.vertices)


@dataclass
class Room:
    """A room defined by polygon vertices (any shape)."""
    name: str
    vertices: List[Tuple[float, float]]
    doors: List[Opening] = field(default_factory=list)
    windows: List[Opening] = field(default_factory=list)
    fill_color: Optional[str] = None
    label: Optional[LabelStyle] = None
    
    @property
    def area(self) -> float:
        return shoelace_area(self.vertices)
    
    @property
    def centroid(self) -> Tuple[float, float]:
        return polygon_centroid(self.vertices)


@dataclass
class VerticalElement:
    """An element spanning multiple floors (elevator, chimney, lightwell, etc.)."""
    type: str  # "elevator", "chimney", "lightwell", "shaft", "dumbwaiter"
    name: str
    vertices: List[Tuple[float, float]]  # polygon defining the footprint
    floors: List[int]  # which floor levels this appears on
    hidden: bool = False  # if True, rendered as "secret/hidden"
    label: Optional[LabelStyle] = None
    
    @property
    def area(self) -> float:
        return shoelace_area(self.vertices)
    
    @property
    def centroid(self) -> Tuple[float, float]:
        return polygon_centroid(self.vertices)


@dataclass
class AlignmentGuide:
    """A guide line for checking alignment across floors."""
    name: str
    points: List[Tuple[float, float]]  # polyline (2+ points)
    floors: List[int]  # which floors to show this on


@dataclass
class Floor:
    """A single floor of the building."""
    level: int
    name: str
    rooms: List[Room] = field(default_factory=list)
    stairs: List[Stairs] = field(default_factory=list)
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) for all rooms."""
        all_vertices = []
        for room in self.rooms:
            all_vertices.extend(room.vertices)
        for stair in self.stairs:
            all_vertices.extend(stair.vertices)
        
        if not all_vertices:
            return (0, 0, 10, 10)
        
        xs = [v[0] for v in all_vertices]
        ys = [v[1] for v in all_vertices]
        return (min(xs), min(ys), max(xs), max(ys))
    
    @property 
    def total_area(self) -> float:
        return sum(room.area for room in self.rooms)


@dataclass
class Building:
    """Complete building definition."""
    name: str
    floors: List[Floor]
    vertical_elements: List[VerticalElement] = field(default_factory=list)
    alignment_guides: List[AlignmentGuide] = field(default_factory=list)
    label_defaults: Optional[LabelStyle] = None  # global label defaults
    
    @property
    def global_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounds encompassing all floors and vertical elements."""
        all_bounds = [f.bounds for f in self.floors]
        
        for ve in self.vertical_elements:
            all_bounds.append(polygon_bounds(ve.vertices))
        
        if not all_bounds:
            return (0, 0, 10, 10)
        
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        return (min_x, min_y, max_x, max_y)
    
    def get_floor_by_level(self, level: int) -> Optional[Floor]:
        """Get floor by its level number."""
        for f in self.floors:
            if f.level == level:
                return f
        return None
    
    def get_vertical_elements_for_floor(self, level: int) -> List[VerticalElement]:
        """Get all vertical elements that appear on a given floor."""
        return [ve for ve in self.vertical_elements if level in ve.floors]
    
    def get_stairs_for_floor(self, level: int) -> List[Stairs]:
        """Get all stairs that connect to a given floor."""
        result = []
        for floor in self.floors:
            for stair in floor.stairs:
                if level in stair.connects:
                    result.append(stair)
        return result
    
    def get_guides_for_floor(self, level: int) -> List[AlignmentGuide]:
        """Get alignment guides for a given floor."""
        return [g for g in self.alignment_guides if level in g.floors]


# =============================================================================
# JSON Parser
# =============================================================================

def parse_label_style(data: Optional[Dict[str, Any]]) -> Optional[LabelStyle]:
    """Parse label style from JSON."""
    if data is None:
        return None
    
    return LabelStyle(
        font=data.get("font"),
        size=data.get("size"),
        color=data.get("color"),
        show_area=data.get("show_area", True),
        show_name=data.get("show_name", True),
        offset_x=data.get("offset_x", 0.0),
        offset_y=data.get("offset_y", 0.0)
    )


def parse_opening(data: Dict[str, Any], opening_type: str) -> Opening:
    """Parse a door or window from JSON."""
    return Opening(
        wall=data.get("wall", 0),
        position=data.get("position", 0.5),
        width=data.get("width", 0.9 if opening_type == "door" else 1.2),
        type=opening_type
    )


def parse_room(data: Dict[str, Any]) -> Room:
    """Parse a room from JSON."""
    vertices = [tuple(v) for v in data.get("vertices", [])]
    doors = [parse_opening(d, "door") for d in data.get("doors", [])]
    windows = [parse_opening(w, "window") for w in data.get("windows", [])]
    
    return Room(
        name=data.get("name", "Unnamed"),
        vertices=vertices,
        doors=doors,
        windows=windows,
        fill_color=data.get("fill_color"),
        label=parse_label_style(data.get("label"))
    )


def parse_stairs(data: Dict[str, Any], floor_level: int) -> Stairs:
    """Parse stairs from JSON."""
    vertices = [tuple(v) for v in data.get("vertices", [])]
    
    # Handle both old format (direction) and new format (connects)
    if "connects" in data:
        connects = tuple(data["connects"])
    else:
        # Legacy: infer from direction
        direction = data.get("direction", "up")
        if direction == "up":
            connects = (floor_level, floor_level + 1)
        else:
            connects = (floor_level - 1, floor_level)
    
    return Stairs(
        vertices=vertices,
        connects=connects,
        name=data.get("name", data.get("label", "")),
        bidirectional=data.get("bidirectional", True),
        label=parse_label_style(data.get("label"))
    )


def parse_floor(data: Dict[str, Any]) -> Floor:
    """Parse a floor from JSON."""
    level = data.get("level", 0)
    rooms = [parse_room(r) for r in data.get("rooms", [])]
    stairs = [parse_stairs(s, level) for s in data.get("stairs", [])]
    
    return Floor(
        level=level,
        name=data.get("name", f"Floor {level}"),
        rooms=rooms,
        stairs=stairs
    )


def parse_vertical_element(data: Dict[str, Any]) -> VerticalElement:
    """Parse a vertical element from JSON."""
    vertices = [tuple(v) for v in data.get("vertices", [])]
    
    return VerticalElement(
        type=data.get("type", "shaft"),
        name=data.get("name", "Unnamed"),
        vertices=vertices,
        floors=data.get("floors", []),
        hidden=data.get("hidden", False),
        label=parse_label_style(data.get("label"))
    )


def parse_alignment_guide(data: Dict[str, Any]) -> AlignmentGuide:
    """Parse an alignment guide from JSON."""
    points = [tuple(p) for p in data.get("points", [])]
    
    return AlignmentGuide(
        name=data.get("name", ""),
        points=points,
        floors=data.get("floors", [])
    )


def parse_building(data: Dict[str, Any]) -> Building:
    """Parse complete building from JSON."""
    floors = [parse_floor(f) for f in data.get("floors", [])]
    floors.sort(key=lambda f: f.level)
    
    vertical_elements = [parse_vertical_element(ve) 
                        for ve in data.get("vertical_elements", [])]
    
    # Legacy support: convert old "secret_shaft" to vertical_element
    if "secret_shaft" in data:
        shaft = data["secret_shaft"]
        x, y = shaft.get("x", 0), shaft.get("y", 0)
        w, d = shaft.get("width", 1.5), shaft.get("depth", 1.5)
        vertices = [(x, y), (x+w, y), (x+w, y+d), (x, y+d)]
        all_levels = [f.level for f in floors]
        vertical_elements.append(VerticalElement(
            type="elevator",
            name=shaft.get("label", "Secret Shaft"),
            vertices=vertices,
            floors=all_levels,
            hidden=True,
            label=parse_label_style(shaft.get("label_style"))
        ))
    
    alignment_guides = [parse_alignment_guide(g) 
                       for g in data.get("alignment_guides", [])]
    
    # Parse global label defaults
    label_defaults = parse_label_style(data.get("label_defaults"))
    
    return Building(
        name=data.get("name", "Unnamed Building"),
        floors=floors,
        vertical_elements=vertical_elements,
        alignment_guides=alignment_guides,
        label_defaults=label_defaults
    )


def load_building(path: str) -> Building:
    """Load building definition from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return parse_building(data)


# =============================================================================
# SVG Generator
# =============================================================================

class SVGGenerator:
    """Generates SVG output from building data."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.elements: List[str] = []
        self.building: Optional[Building] = None  # set during generate()
    
    def _resolve_label_style(self, element_label: Optional[LabelStyle],
                             default_color: str) -> Tuple[str, float, str]:
        """Resolve label font, size, color from element + building defaults + config.
        
        Returns (font, size, color).
        """
        # Start with config defaults
        font = self.config.label_font
        size = self.config.label_font_size
        color = default_color
        
        # Apply building-level defaults
        if self.building and self.building.label_defaults:
            bd = self.building.label_defaults
            if bd.font:
                font = bd.font
            if bd.size:
                size = bd.size if bd.size >= 5 else self.config.label_font_size * bd.size
            if bd.color:
                color = bd.color
        
        # Apply element-level overrides
        if element_label:
            if element_label.font:
                font = element_label.font
            if element_label.size:
                size = element_label.size if element_label.size >= 5 else self.config.label_font_size * element_label.size
            if element_label.color:
                color = element_label.color
        
        return (font, size, color)
    
    def _transform_point(self, x: float, y: float, 
                         offset_x: float, offset_y: float) -> Tuple[float, float]:
        """Transform building coordinates to SVG coordinates."""
        return (
            (x + offset_x) * self.config.scale,
            (y + offset_y) * self.config.scale
        )
    
    def _polygon_path(self, vertices: List[Tuple[float, float]], 
                      offset_x: float, offset_y: float) -> str:
        """Generate SVG path data for a polygon."""
        if not vertices:
            return ""
        
        points = [self._transform_point(v[0], v[1], offset_x, offset_y) 
                  for v in vertices]
        
        d = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        for p in points[1:]:
            d += f" L {p[0]:.2f},{p[1]:.2f}"
        d += " Z"
        return d
    
    def _polyline_path(self, points: List[Tuple[float, float]],
                       offset_x: float, offset_y: float) -> str:
        """Generate SVG path data for a polyline (not closed)."""
        if not points:
            return ""
        
        transformed = [self._transform_point(p[0], p[1], offset_x, offset_y)
                      for p in points]
        
        d = f"M {transformed[0][0]:.2f},{transformed[0][1]:.2f}"
        for p in transformed[1:]:
            d += f" L {p[0]:.2f},{p[1]:.2f}"
        return d
    
    def _draw_room_fill(self, room: Room, offset_x: float, offset_y: float) -> str:
        """Draw room fill (light background)."""
        path = self._polygon_path(room.vertices, offset_x, offset_y)
        fill_color = room.fill_color or self.config.wall_color
        return f'<path d="{path}" fill="{fill_color}" fill-opacity="{self.config.room_fill_opacity}" stroke="none"/>'
    
    def _draw_walls(self, room: Room, offset_x: float, offset_y: float) -> List[str]:
        """Draw walls with gaps for doors and windows."""
        elements = []
        vertices = room.vertices
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            p1, p2 = vertices[i], vertices[j]
            
            # Find all openings on this wall
            openings = []
            for door in room.doors:
                if door.wall == i:
                    openings.append(door)
            for window in room.windows:
                if window.wall == i:
                    openings.append(window)
            
            if not openings:
                # Draw full wall
                sp1 = self._transform_point(p1[0], p1[1], offset_x, offset_y)
                sp2 = self._transform_point(p2[0], p2[1], offset_x, offset_y)
                elements.append(
                    f'<line x1="{sp1[0]:.2f}" y1="{sp1[1]:.2f}" '
                    f'x2="{sp2[0]:.2f}" y2="{sp2[1]:.2f}" '
                    f'stroke="{self.config.wall_color}" '
                    f'stroke-width="{self.config.wall_width}" '
                    f'stroke-linecap="round"/>'
                )
            else:
                # Sort openings by position
                openings.sort(key=lambda o: o.position)
                wall_len = segment_length(p1, p2)
                
                # Draw wall segments between openings
                segments = []
                current_t = 0.0
                
                for opening in openings:
                    half_width_t = (opening.width / 2) / wall_len if wall_len > 0 else 0
                    start_t = opening.position - half_width_t
                    end_t = opening.position + half_width_t
                    
                    if start_t > current_t:
                        segments.append((current_t, start_t))
                    current_t = max(current_t, end_t)
                
                if current_t < 1.0:
                    segments.append((current_t, 1.0))
                
                for seg_start, seg_end in segments:
                    seg_p1 = point_on_segment(p1, p2, seg_start)
                    seg_p2 = point_on_segment(p1, p2, seg_end)
                    sp1 = self._transform_point(seg_p1[0], seg_p1[1], offset_x, offset_y)
                    sp2 = self._transform_point(seg_p2[0], seg_p2[1], offset_x, offset_y)
                    elements.append(
                        f'<line x1="{sp1[0]:.2f}" y1="{sp1[1]:.2f}" '
                        f'x2="{sp2[0]:.2f}" y2="{sp2[1]:.2f}" '
                        f'stroke="{self.config.wall_color}" '
                        f'stroke-width="{self.config.wall_width}" '
                        f'stroke-linecap="round"/>'
                    )
                
                # Draw the openings
                for opening in openings:
                    wall_len = segment_length(p1, p2)
                    half_width_t = (opening.width / 2) / wall_len if wall_len > 0 else 0
                    start_t = opening.position - half_width_t
                    end_t = opening.position + half_width_t
                    
                    op1 = point_on_segment(p1, p2, max(0, start_t))
                    op2 = point_on_segment(p1, p2, min(1, end_t))
                    
                    if opening.type == "door":
                        sp1 = self._transform_point(op1[0], op1[1], offset_x, offset_y)
                        sp2 = self._transform_point(op2[0], op2[1], offset_x, offset_y)
                        elements.append(
                            f'<line x1="{sp1[0]:.2f}" y1="{sp1[1]:.2f}" '
                            f'x2="{sp2[0]:.2f}" y2="{sp2[1]:.2f}" '
                            f'stroke="{self.config.door_color}" '
                            f'stroke-width="{self.config.door_width}" '
                            f'stroke-linecap="round"/>'
                        )
                    else:  # window
                        sp1 = self._transform_point(op1[0], op1[1], offset_x, offset_y)
                        sp2 = self._transform_point(op2[0], op2[1], offset_x, offset_y)
                        elements.append(
                            f'<line x1="{sp1[0]:.2f}" y1="{sp1[1]:.2f}" '
                            f'x2="{sp2[0]:.2f}" y2="{sp2[1]:.2f}" '
                            f'stroke="{self.config.window_color}" '
                            f'stroke-width="{self.config.window_width}" '
                            f'stroke-linecap="round"/>'
                        )
        
        return elements
    
    def _draw_stairs(self, stairs: Stairs, floor_level: int,
                     offset_x: float, offset_y: float) -> List[str]:
        """Draw stairs with connection info."""
        elements = []
        
        if len(stairs.vertices) < 3:
            return elements
        
        # Draw outline
        path = self._polygon_path(stairs.vertices, offset_x, offset_y)
        elements.append(
            f'<path d="{path}" fill="{self.config.stairs_color}" '
            f'fill-opacity="0.1" stroke="{self.config.stairs_color}" '
            f'stroke-width="{self.config.stairs_width}"/>'
        )
        
        # Draw tread lines inside (works for any quadrilateral)
        if len(stairs.vertices) >= 4:
            p0, p1, p2, p3 = stairs.vertices[:4]
            
            len01 = segment_length(p0, p1)
            len12 = segment_length(p1, p2)
            
            if len01 > len12:
                start_edge = (p0, p3)
                end_edge = (p1, p2)
                num_treads = max(2, int(len01 / 0.3))
            else:
                start_edge = (p0, p1)
                end_edge = (p3, p2)
                num_treads = max(2, int(len12 / 0.3))
            
            for i in range(1, num_treads):
                t = i / num_treads
                tp1 = point_on_segment(start_edge[0], end_edge[0], t)
                tp2 = point_on_segment(start_edge[1], end_edge[1], t)
                sp1 = self._transform_point(tp1[0], tp1[1], offset_x, offset_y)
                sp2 = self._transform_point(tp2[0], tp2[1], offset_x, offset_y)
                elements.append(
                    f'<line x1="{sp1[0]:.2f}" y1="{sp1[1]:.2f}" '
                    f'x2="{sp2[0]:.2f}" y2="{sp2[1]:.2f}" '
                    f'stroke="{self.config.stairs_color}" '
                    f'stroke-width="1" stroke-opacity="0.5"/>'
                )
        
        # Direction label showing connection
        label_style = stairs.label or LabelStyle()
        
        if label_style.show_name:
            centroid = polygon_centroid(stairs.vertices)
            cx = centroid[0] + label_style.offset_x
            cy = centroid[1] + label_style.offset_y
            sc = self._transform_point(cx, cy, offset_x, offset_y)
            
            lower, upper = stairs.connects
            if floor_level == lower:
                arrow = "↑"
                target = upper
            else:
                arrow = "↓"
                target = lower
            
            label = stairs.name if stairs.name else f"{arrow} L{target}"
            
            font, size, color = self._resolve_label_style(stairs.label, self.config.stairs_color)
            
            elements.append(
                f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                f'font-family="{font}" font-size="{size}" '
                f'fill="{color}" text-anchor="middle" '
                f'dominant-baseline="middle">{label}</text>'
            )
        
        return elements
    
    def _draw_vertical_element(self, ve: VerticalElement, floor_level: int,
                               offset_x: float, offset_y: float) -> List[str]:
        """Draw a vertical element (elevator, chimney, etc.)."""
        elements = []
        
        # Select color based on type and hidden status
        if ve.hidden:
            element_color = self.config.elevator_hidden_color
            dash = self.config.vertical_hidden_dash
        else:
            color_map = {
                "elevator": self.config.elevator_color,
                "chimney": self.config.chimney_color,
                "lightwell": self.config.lightwell_color,
                "shaft": self.config.shaft_color,
                "dumbwaiter": self.config.elevator_color,
            }
            element_color = color_map.get(ve.type, self.config.shaft_color)
            dash = self.config.vertical_dash
        
        # Draw polygon
        path = self._polygon_path(ve.vertices, offset_x, offset_y)
        elements.append(
            f'<path d="{path}" fill="{element_color}" fill-opacity="0.15" '
            f'stroke="{element_color}" stroke-width="{self.config.vertical_width}" '
            f'stroke-dasharray="{dash}"/>'
        )
        
        # Label
        label_style = ve.label or LabelStyle()
        
        if label_style.show_name or label_style.show_area:
            cx, cy = ve.centroid
            cx += label_style.offset_x
            cy += label_style.offset_y
            sc = self._transform_point(cx, cy, offset_x, offset_y)
            
            font, size, color = self._resolve_label_style(ve.label, element_color)
            # Vertical elements use slightly smaller text by default
            size = size * 0.8
            
            # Show floor range
            floor_range = f"L{min(ve.floors)}–{max(ve.floors)}" if len(ve.floors) > 1 else ""
            
            if label_style.show_name and label_style.show_area:
                elements.append(
                    f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                    f'font-family="{font}" font-size="{size:.1f}" '
                    f'fill="{color}" text-anchor="middle" font-style="italic">'
                    f'<tspan x="{sc[0]:.2f}" dy="-0.3em">{ve.name}</tspan>'
                    f'<tspan x="{sc[0]:.2f}" dy="1.1em" font-size="{size*0.85:.1f}">'
                    f'{ve.area:.1f}m² {floor_range}</tspan></text>'
                )
            elif label_style.show_name:
                elements.append(
                    f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                    f'font-family="{font}" font-size="{size:.1f}" '
                    f'fill="{color}" text-anchor="middle" font-style="italic" '
                    f'dominant-baseline="middle">{ve.name}</text>'
                )
            else:  # show_area only
                elements.append(
                    f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                    f'font-family="{font}" font-size="{size*0.85:.1f}" '
                    f'fill="{color}" text-anchor="middle" font-style="italic" '
                    f'dominant-baseline="middle">{ve.area:.1f}m²</text>'
                )
        
        return elements
    
    def _draw_alignment_guide(self, guide: AlignmentGuide,
                              offset_x: float, offset_y: float) -> List[str]:
        """Draw an alignment guide line."""
        elements = []
        
        if len(guide.points) < 2:
            return elements
        
        path = self._polyline_path(guide.points, offset_x, offset_y)
        elements.append(
            f'<path d="{path}" fill="none" '
            f'stroke="{self.config.guide_color}" '
            f'stroke-width="{self.config.guide_width}" '
            f'stroke-dasharray="{self.config.guide_dash}" '
            f'stroke-opacity="0.7"/>'
        )
        
        # Label at midpoint
        if guide.name:
            mid_idx = len(guide.points) // 2
            mx, my = guide.points[mid_idx]
            sm = self._transform_point(mx, my, offset_x, offset_y)
            elements.append(
                f'<text x="{sm[0]:.2f}" y="{sm[1] - 5:.2f}" '
                f'font-family="Arial, sans-serif" font-size="{self.config.label_font_size * 0.7:.1f}" '
                f'fill="{self.config.guide_color}" text-anchor="middle" '
                f'font-style="italic">{guide.name}</text>'
            )
        
        return elements
    
    def _draw_room_label(self, room: Room, offset_x: float, offset_y: float) -> str:
        """Draw room label with name and area."""
        label_style = room.label or LabelStyle()
        
        # Check if we should show anything
        if not label_style.show_name and not label_style.show_area:
            return ""
        
        cx, cy = room.centroid
        # Apply offset
        cx += label_style.offset_x
        cy += label_style.offset_y
        
        sc = self._transform_point(cx, cy, offset_x, offset_y)
        area = room.area
        
        # Resolve font, size, color
        font, size, color = self._resolve_label_style(room.label, self.config.wall_color)
        
        # Build label text
        if label_style.show_name and label_style.show_area:
            return (
                f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                f'font-family="{font}" font-size="{size}" '
                f'fill="{color}" text-anchor="middle">'
                f'<tspan x="{sc[0]:.2f}" dy="-0.4em">{room.name}</tspan>'
                f'<tspan x="{sc[0]:.2f}" dy="1.2em" font-size="{size * 0.85:.1f}">'
                f'{area:.1f} m²</tspan></text>'
            )
        elif label_style.show_name:
            return (
                f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                f'font-family="{font}" font-size="{size}" '
                f'fill="{color}" text-anchor="middle" dominant-baseline="middle">'
                f'{room.name}</text>'
            )
        else:  # show_area only
            return (
                f'<text x="{sc[0]:.2f}" y="{sc[1]:.2f}" '
                f'font-family="{font}" font-size="{size * 0.85:.1f}" '
                f'fill="{color}" text-anchor="middle" dominant-baseline="middle">'
                f'{area:.1f} m²</text>'
            )
    
    def _draw_floor(self, floor: Floor, building: Building,
                    offset_x: float, offset_y: float,
                    bounds: Tuple[float, float, float, float]) -> List[str]:
        """Draw a complete floor with all elements."""
        elements = []
        level = floor.level
        
        # Floor background
        padding = self.config.floor_padding
        bx1, by1, bx2, by2 = bounds
        sp1 = self._transform_point(bx1 - padding, by1 - padding, offset_x, offset_y)
        sp2 = self._transform_point(bx2 + padding, by2 + padding, offset_x, offset_y)
        elements.append(
            f'<rect x="{sp1[0]:.2f}" y="{sp1[1]:.2f}" '
            f'width="{sp2[0] - sp1[0]:.2f}" height="{sp2[1] - sp1[1]:.2f}" '
            f'fill="{self.config.background_color}" stroke="#ddd" stroke-width="1"/>'
        )
        
        # Floor title with total area
        title_pos = self._transform_point(bx1 - padding + 0.3, by1 - padding + 0.5,
                                          offset_x, offset_y)
        total_area = floor.total_area
        elements.append(
            f'<text x="{title_pos[0]:.2f}" y="{title_pos[1]:.2f}" '
            f'font-family="{self.config.label_font}" font-size="{self.config.floor_label_font_size}" '
            f'fill="{self.config.wall_color}" font-weight="bold">'
            f'Level {level}: {floor.name}'
            f'<tspan font-weight="normal" font-size="{self.config.floor_label_font_size * 0.75:.1f}"> '
            f'({total_area:.0f} m²)</tspan></text>'
        )
        
        # Alignment guides (bottom layer)
        for guide in building.get_guides_for_floor(level):
            elements.extend(self._draw_alignment_guide(guide, offset_x, offset_y))
        
        # Room fills
        for room in floor.rooms:
            elements.append(self._draw_room_fill(room, offset_x, offset_y))
        
        # Walls
        for room in floor.rooms:
            elements.extend(self._draw_walls(room, offset_x, offset_y))
        
        # Stairs (those that connect to this floor)
        drawn_stairs: Set[int] = set()
        for stair in floor.stairs:
            if level in stair.connects:
                stair_id = id(stair)
                if stair_id not in drawn_stairs:
                    elements.extend(self._draw_stairs(stair, level, offset_x, offset_y))
                    drawn_stairs.add(stair_id)
        
        # Vertical elements
        for ve in building.get_vertical_elements_for_floor(level):
            elements.extend(self._draw_vertical_element(ve, level, offset_x, offset_y))
        
        # Room labels (on top)
        for room in floor.rooms:
            elements.append(self._draw_room_label(room, offset_x, offset_y))
        
        return elements
    
    def _build_legend(self, building: Building) -> str:
        """Build legend showing all element types present."""
        items = [
            (self.config.wall_color, "Walls"),
            (self.config.door_color, "Doors"),
            (self.config.window_color, "Windows"),
            (self.config.stairs_color, "Stairs"),
        ]
        
        # Add vertical element types that are actually used
        ve_types = set(ve.type for ve in building.vertical_elements)
        hidden_present = any(ve.hidden for ve in building.vertical_elements)
        
        if hidden_present:
            items.append((self.config.elevator_hidden_color, "Hidden Shaft"))
        if "elevator" in ve_types and not all(ve.hidden for ve in building.vertical_elements if ve.type == "elevator"):
            items.append((self.config.elevator_color, "Elevator"))
        if "chimney" in ve_types:
            items.append((self.config.chimney_color, "Chimney"))
        if "lightwell" in ve_types:
            items.append((self.config.lightwell_color, "Lightwell"))
        if "shaft" in ve_types:
            items.append((self.config.shaft_color, "Shaft"))
        
        if building.alignment_guides:
            items.append((self.config.guide_color, "Alignment Guide"))
        
        legend_parts = []
        for color, label in items:
            legend_parts.append(f'<tspan fill="{color}">■</tspan> {label}  ')
        
        return ''.join(legend_parts)
    
    def generate(self, building: Building) -> str:
        """Generate complete SVG for the building."""
        self.building = building  # Store for label resolution
        
        bounds = building.global_bounds
        bx1, by1, bx2, by2 = bounds
        padding = self.config.floor_padding
        
        floor_width = (bx2 - bx1) + 2 * padding
        floor_height = (by2 - by1) + 2 * padding
        spacing = self.config.floor_spacing
        
        num_floors = len(building.floors)
        cols = self.config.grid_columns
        rows = math.ceil(num_floors / cols)
        
        total_width = cols * floor_width + (cols - 1) * spacing
        total_height = rows * floor_height + (rows - 1) * spacing
        
        svg_width = total_width * self.config.scale
        svg_height = total_height * self.config.scale + 30  # Extra for legend
        
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width:.0f}" height="{svg_height:.0f}" '
            f'viewBox="0 0 {svg_width:.0f} {svg_height:.0f}">',
            f'<rect width="100%" height="100%" fill="white"/>',
        ]
        
        # Draw each floor
        for idx, floor in enumerate(building.floors):
            col = idx % cols
            row = idx // cols
            
            offset_x = col * (floor_width + spacing) - bx1 + padding
            offset_y = row * (floor_height + spacing) - by1 + padding
            
            floor_elements = self._draw_floor(floor, building, offset_x, offset_y, bounds)
            svg_parts.extend(floor_elements)
        
        # Legend
        legend_y = total_height * self.config.scale + 15
        legend_text = self._build_legend(building)
        svg_parts.append(
            f'<g transform="translate(20, {legend_y})">'
            f'<text font-family="Arial, sans-serif" font-size="10" fill="#666">'
            f'{legend_text}</text></g>'
        )
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)


# =============================================================================
# Summary Report
# =============================================================================

def print_summary(building: Building):
    """Print a summary of the building."""
    print(f"\nBuilding: {building.name}")
    print(f"{'='*50}")
    
    total_building_area = 0
    
    for floor in building.floors:
        floor_area = floor.total_area
        total_building_area += floor_area
        print(f"\nLevel {floor.level}: {floor.name}")
        print(f"  Rooms: {len(floor.rooms)}, Total area: {floor_area:.1f} m²")
        
        for room in floor.rooms:
            print(f"    - {room.name}: {room.area:.1f} m²")
        
        if floor.stairs:
            print(f"  Stairs:")
            for stair in floor.stairs:
                lower, upper = stair.connects
                print(f"    - {stair.name or 'Unnamed'}: L{lower} ↔ L{upper} ({stair.area:.1f} m²)")
    
    if building.vertical_elements:
        print(f"\nVertical Elements:")
        for ve in building.vertical_elements:
            floors_str = ", ".join(str(f) for f in sorted(ve.floors))
            hidden_str = " [HIDDEN]" if ve.hidden else ""
            print(f"  - {ve.name} ({ve.type}): {ve.area:.1f} m² on floors {floors_str}{hidden_str}")
            
            # Calculate percentage of floor area
            for floor in building.floors:
                if floor.level in ve.floors and floor.total_area > 0:
                    pct = (ve.area / floor.total_area) * 100
                    print(f"      L{floor.level}: {pct:.2f}% of floor area")
    
    if building.alignment_guides:
        print(f"\nAlignment Guides:")
        for guide in building.alignment_guides:
            floors_str = ", ".join(str(f) for f in sorted(guide.floors))
            print(f"  - {guide.name}: floors {floors_str}")
    
    print(f"\n{'='*50}")
    print(f"Total building area: {total_building_area:.1f} m²")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python floorplan.py <input.json> [output.svg]")
        print("\nGenerates SVG floorplan visualization from JSON definition.")
        print("\nFeatures:")
        print("  - Multi-floor visualization with vertical alignment")
        print("  - Vertical elements (elevators, chimneys) spanning multiple floors")
        print("  - Arbitrary polygon room shapes")
        print("  - Auto-calculated room areas")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.rsplit('.', 1)[0] + '.svg'
    
    print(f"Loading building from {input_path}...")
    building = load_building(input_path)
    
    print_summary(building)
    
    print(f"\nGenerating SVG...")
    generator = SVGGenerator()
    svg_content = generator.generate(building)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
