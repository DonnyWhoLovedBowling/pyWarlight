# from svgelements import SVGElement, Path, Point, Matrix, Line


def to_global(element: SVGElement, point: Point):
    current_element = element
    global_point = Point(point.x, point.y)

    while True:
        if hasattr(current_element, 'parent'):
            parent = current_element.parent
        else:
            parent = None

        if parent is None:
            break

        if hasattr(parent, 'transform') and parent.transform is not None:
            matrix = Matrix(parent.transform)
            global_point *= matrix

        current_element = parent

    return global_point



def on_segment(p, q, r):
    """Checks if point q lies on line segment pr."""
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

def orientation(p, q, r):
    """
    Finds the orientation of ordered triplet (p, q, r).
    Returns 0 if p, q, r are collinear,
            1 if clockwise,
            2 if counterclockwise.
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def do_intersect(p1, q1, p2, q2):
    """Checks if line segment p1q1 intersects with line segment p2q2."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special Cases for collinear points
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def is_point_in_path(path: Path, point: Point) -> bool:
    """
    Checks if a Point is inside a Path using the Ray Casting Algorithm.

    Args:
        path (Path): The svgelements Path object.
        point (Point): The svgelements Point object.

    Returns:
        bool: True if the point is inside the path, False otherwise.
    """
    intersections = 0
    n = len(path)
    if n < 3:  # Need at least 3 points for a polygon
        return False

    ray_end = Point(float('inf'), point.y)  # Cast a horizontal ray to the right

    for i in range(n):
        p1 = path[i].end if hasattr(path[i], 'end') else path[i][1] if len(path[i]) > 1 else None
        p2 = path[(i + 1) % n].end if hasattr(path[(i + 1) % n], 'end') else path[(i + 1) % n][1] if len(path[(i + 1) % n]) > 1 else None

        if p1 is None or p2 is None:
            continue

        # Handle different segment types (assuming linear segments for simplicity)
        if isinstance(path[i], Line) and isinstance(path[(i + 1) % n], Line):
            if do_intersect(p1, p2, point, ray_end):
                # If the point lies on the segment, consider it inside (you might adjust this)
                if orientation(p1, point, p2) == 0 and on_segment(p1, point, p2):
                    return True
                intersections += 1
        # You would need to add more complex logic to handle Curve, QuadraticBezier, etc.
        # by approximating them with line segments or using more advanced intersection tests.

    return intersections % 2 == 1

