"""
Smart Study Desk - ONE merged 3D printable enclosure (parametric)

Top view (front = user side, -Y):

   +-----------------------------------------------+  back
   |  PHONE compartment (sliding lid, slides out ->)|  z 42..66
   |  POWER BANK bay: 2 slots stacked, open L & R   |  z  2..40   (under the phone)
   |  --- guard rib ---------------------------------|
   |  SERVO SWING strip 15 mm (arm clearance >= 5)   |
   |=================== divider =====================|  servo passes through here
   |   ESP32-S3 zone (Sensor Node)  |  Raspberry Pi 5 |
   |   HY-SRF05 + 5V fan on front   |  zone (Gateway) |
   +-----------------------------------------------+  front

ESP32-S3 zone : ESP32-S3 DevKitC, KY-015 (DHT11), KY-018 (LDR), MQ135, HY-SRF05,
                5V fan + MOSFET (on mini breadboard), SG90 servo (phone lock), 5V input for servo/fan
Pi zone       : Raspberry Pi 5, Pi Camera (CSI) on tilted mount, KY-006 buzzer
Power         : 2 power banks (bank 1 -> ESP32 + 5V rail, bank 2 -> Pi 5 via the Pi power hole)

Run:  python build.py
Out:  stl/*.stl, renders/*.png, viewer.html (hover a part to see what it is)
Requires: pip install manifold3d trimesh numpy matplotlib
Units: mm.  X = width, Y = depth (-Y = front), Z = up.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold, OpType

HERE = Path(__file__).resolve().parent
STL_DIR = HERE / "stl"
RENDER_DIR = HERE / "renders"

# ============================================================== overall layout
W, H = 192.0, 71.0
WALL, FLOOR = 2.0, 2.0
LID_T, LIP_H, LIP_T, FIT = 2.0, 4.0, 1.6, 0.2
CORNER_R = 4.0
SEG = 64
XI = W / 2 - WALL                       # 94: inner half width

FRONT_DEPTH = 92.0                      # electronics section (inner)
DIV_T = 4.0                             # divider wall (servo screws in, phone-lid groove behind)
SWING = 15.0                            # servo swing strip (user request: +1.5 cm)
RIB_T = 2.0                             # guard rib between swing strip and phone
PHONE_DEPTH = 86.0                      # phone compartment / power bank bay (inner)
BACK_T = 3.5                            # back wall (carries the phone-lid groove)
D = WALL + FRONT_DEPTH + DIV_T + SWING + RIB_T + PHONE_DEPTH + BACK_T

Y0 = -D / 2                             # front outer face
YF = Y0 + WALL                          # front inner face
Y_D = YF + FRONT_DEPTH                  # divider front face
YS = Y_D + DIV_T                        # swing strip start (divider back face)
Y_RIB = YS + SWING                      # guard rib front face
YP = Y_RIB + RIB_T                      # phone compartment start
YB = YP + PHONE_DEPTH                   # back inner face

X_DIV = 9.0                             # ESP32 | Pi divider centre
PLUG_GAP = 15.0                         # room inside for a USB-C plug (cable from the power bank)
CH_Z = (3.0, 15.0)                      # cable holes: divider (front <-> channel) height

# back section heights
BANK_H = 18.0                           # slot height (power bank up to ~16-17 mm thick)
Z_SHELF = FLOOR + BANK_H                # 20
Z_PHONE_FLOOR = Z_SHELF + 2 + BANK_H    # 40
Z_PHONE = Z_PHONE_FLOOR + 2             # 42 phone cavity bottom
RIB_TOP = Z_PHONE + 10                  # 52
STRIP_SHELF = 28.0                      # swing strip floor (top at +2)
G0 = H - 5.0                            # phone-lid groove bottom (66)
G1 = G0 + 2.9                           # groove top
PL0 = G0 + 0.3                          # phone lid bottom (66.3)
PL_T = 2.3
GROOVE_D = 1.5
POWERBANK = (145.0, 70.0, 22.0)         # mock-up size (single slot fits ~ 150 x 82 x 37)
# phone floor = SEPARATE printed plate (an 86 mm wide slab over the bay can't be printed without support)
SHELF_LEDGE = 3.0                       # 45 deg ledges on the guard rib + back wall carry the plate
SHELF_FIT = 0.3                         # clearance around the plate
SHELF_HOLE_D = 14.0                     # finger holes to lift the plate out

# ============================================================== servo (SG90)
SV_X = -40.0                            # body centre x (ESP32 zone)
SV_SHAFT_OFF = 5.9                      # shaft offset from body centre along its length
SV_SHAFT_X = SV_X + SV_SHAFT_OFF
SV_L, SV_WID, SV_TAB_L = 22.8, 12.2, 32.3
SV_BODY_H, SV_TAB_POS, SV_TAB_T, SV_HUMP = 22.7, 15.9, 2.5, 4.0
ARM, ARM_W, ARM_T, TIP_IN = 16.0, 5.0, 1.6, 1.5
SV_Z = PL0 + TIP_IN - ARM               # shaft height so the arm tip enters the lid slot
SV_BOTTOM_Y = Y_D - (SV_TAB_POS + SV_TAB_T)   # tabs screwed onto the divider front face
SV_TOP_Y = SV_BOTTOM_Y + SV_BODY_H
HORN_Y = SV_TOP_Y + SV_HUMP + ARM_T     # arm plane centre
SWING_CLEAR = 5.0

# ============================================================== ESP32-S3 zone
HY_CX, HY_Z0 = -64.0, 25.0              # HY-SRF05 (45 x 20, Ø16 transducers, 26 mm spacing)
HY_HOLE_D, HY_SPACING, HY_BOARD = 16.6, 26.0, (45.0, 20.0)
HY_CZ = HY_Z0 + HY_BOARD[1] / 2
HY_PCB_Y = YF + 10.0
FAN, FAN_T, FAN_SCREW, FAN_GRILLE_D = 40.0, 10.0, 32.0, 37.0
FAN_CX, FAN_CZ = -14.0, 35.0
ESP_L, ESP_W = 70.0, 26.5               # ESP32-S3 DevKitC cradle, USB-C end -> left wall
ESP_X0 = -XI + PLUG_GAP                 # USB-C end, PLUG_GAP from the left wall
ESP_CX, ESP_CY = ESP_X0 + ESP_L / 2, -60.0
ESP_PCB_Z = 10.5
ESP_LED = (ESP_X0 + 23.0, ESP_CY + 5.0)  # RGB LED position -> light pipe (check your board!)
DHT_CY, DHT_CZ = -30.0, 32.0            # KY-015 on the left wall
LDR_C, LDR_BOARD = (-72.0, -30.0), (18.9, 15.4)   # KY-018 under the lid, LDR through Ø6
MQ_C, MQ_BOARD, MQ_HOLE_D = (-30.0, -42.0), (32.6, 22.6), 20.6
BB = (47.2, 35.2)                       # mini breadboard (5V rail, MOSFET, divider, capacitor)
BB_C = (-40.0, -26.0)
V5_Y, V5_Z = -12.0, 10.0                # USB-C 5V input (servo + fan) on the left wall

# ============================================================== Pi zone
PI_CASE = (70.0, 90.0, 32.0)           # Raspberry Pi 5 IN ITS CASE: 7 cm wide x 9 cm long (height assumed)
PI_CASE_X1 = XI - 13.0                  # 13 mm gap to the right wall for a right-angle USB-C plug
PI_CASE_X0 = PI_CASE_X1 - PI_CASE[0]
PI_CASE_Y0 = YF + 1.0                   # USB/Ethernet end of the case -> front wall
PI_CASE_Y1 = PI_CASE_Y0 + PI_CASE[1]
PI_X1 = PI_CASE_X1 - (PI_CASE[0] - 56.0) / 2   # board inside the case (USB-C/HDMI side -> right)
PI_X0 = PI_X1 - 56.0
PI_Y0 = PI_CASE_Y0 + (PI_CASE[1] - 85.0) / 2
PI_Y1 = PI_Y0 + 85.0                    # microSD end
PI_PCB_Z = FLOOR + 4.0                  # board height inside the case (approx.)
PI_USBC_Y = PI_Y1 - 11.2
PI_HDMI_Y = (PI_Y1 - 26.0, PI_Y1 - 39.5)
BUZ_C, BUZ_D = (74.0, -28.0), 12.4      # KY-006 under the lid

CAM_C = (55.0, -78.0)                   # camera zone on the lid
CAM_BASE = (34.0, 30.0)
CAM_TILT = 25.0
CAM_PLATE = (32.0, 3.0, 32.0)
CAM_HOLE_DX, CAM_HOLE_Z, CAM_HOLE_DZ = 10.5, 6.0, 12.5
CAM_LENS = (10.0, 14.0)
CAM_FIX = [(-12.0, 9.0), (12.0, 9.0)]
CAM_RIBBON = (18.0, 3.0, 4.0)


# ============================================================== helpers
def box(x, y, z, cx=0.0, cy=0.0, z0=0.0) -> Manifold:
    return Manifold.cube((x, y, z), center=True).translate((cx, cy, z0 + z / 2))


def span(x0, x1, y0, y1, z0, z1) -> Manifold:
    return box(x1 - x0, y1 - y0, z1 - z0, (x0 + x1) / 2, (y0 + y1) / 2, z0)


def rbox(x, y, z, r, cx=0.0, cy=0.0, z0=0.0) -> Manifold:
    r = max(min(r, x / 2 - 0.01, y / 2 - 0.01), 0.01)
    cs = CrossSection.square((x - 2 * r, y - 2 * r), center=True).offset(r, JoinType.Round, 2.0, SEG)
    return cs.extrude(z).translate((cx, cy, z0))


def cyl_z(d, h, cx=0.0, cy=0.0, z0=0.0, d2=None) -> Manifold:
    return Manifold.cylinder(h, d / 2, -1.0 if d2 is None else d2 / 2, SEG).translate((cx, cy, z0))


def cyl_y(d, length, cx, cy, cz) -> Manifold:
    return Manifold.cylinder(length, d / 2, -1.0, SEG, True).rotate((90, 0, 0)).translate((cx, cy, cz))


def cyl_x(d, length, cx, cy, cz) -> Manifold:
    return Manifold.cylinder(length, d / 2, -1.0, SEG, True).rotate((0, 90, 0)).translate((cx, cy, cz))


def cyl_y_span(d, y0, y1, cx, cz) -> Manifold:
    return cyl_y(d, abs(y1 - y0), cx, (y0 + y1) / 2, cz)


def union(parts) -> Manifold:
    parts = list(parts)
    return Manifold.batch_boolean(parts, OpType.Add) if parts else Manifold()


def frame(inner_x, inner_y, h, t, cx, cy, ztop) -> Manifold:
    return box(inner_x + 2 * t, inner_y + 2 * t, h, cx, cy, ztop - h) - box(
        inner_x, inner_y, h + 1, cx, cy, ztop - h - 0.5)


def pocket(inner_x, inner_y, h, cx, cy) -> Manifold:
    return box(inner_x + 2.4, inner_y + 2.4, h, cx, cy, FLOOR) - box(inner_x, inner_y, h + 2, cx, cy, FLOOR - 1)


def flip_to_bed(m: Manifold) -> Manifold:
    m = m.rotate((180, 0, 0))
    return m.translate((0, 0, -m.bounding_box()[2]))


def to_bed(m: Manifold) -> Manifold:
    return m.translate((0, 0, -m.bounding_box()[2]))


# ============================================================== BODY (one piece)
def body() -> Manifold:
    m = rbox(W, D, H, CORNER_R)
    m = m - span(-XI, XI, YF, Y_D, FLOOR, H + 1)          # electronics section
    m = m - span(-XI, XI, YS, YB, FLOOR, H + 1)           # swing strip + phone + bay
    add, cut = [], []

    # ---- back section internals
    add.append(span(-XI - 0.5, XI + 0.5, YS - 0.5, Y_RIB + 0.01, STRIP_SHELF, STRIP_SHELF + 2))  # strip floor
    add.append(span(-XI - 0.5, XI + 0.5, Y_RIB, YP, FLOOR - 0.5, RIB_TOP))                         # guard rib
    for z0, z1 in ((FLOOR + 0.5, Z_PHONE_FLOOR - 0.5),):                                           # bank opening (1 slot)
        cut.append(span(-W, W, YP + 2.0, YB - 2.0, z0, z1))
    # cable channel: power bank bay -> (rib holes) -> space under the swing strip -> (divider holes) -> ESP32 / Pi
    for sx in (-1, 1):
        x = sx * (XI - 12.0)
        for z0, z1 in ((FLOOR + 2.0, STRIP_SHELF - 0.5),):
            cut.append(span(x - 8.0, x + 8.0, Y_RIB - 1, YP + 1, z0, z1))
    cut.append(span(-XI + 2.0, -XI + 16.0, Y_D - 1, YS + 1, *CH_Z))      # -> ESP32 zone
    # phone-lid grooves (divider back face + back wall) and exit slot in the right wall
    cut.append(span(-XI, W, YS - GROOVE_D, YS + 0.01, G0, G1))
    cut.append(span(-XI, W, YB - 0.01, YB + GROOVE_D, G0, G1))
    cut.append(span(XI - 1, W, YS - GROOVE_D, YB + GROOVE_D, G0, H + 1))

    # ---- ESP32 | Pi divider (with a cable notch)
    add.append(span(X_DIV - 1, X_DIV + 1, YF - 0.5, Y_D + 0.5, FLOOR - 0.5, H - LIP_H - 0.5))
    cut.append(span(X_DIV - 2, X_DIV + 2, -40.0, -28.0, H - LIP_H - 9, H))

    # ---- servo through the divider: case opening + 2 screw holes (M2) for the tabs
    cut.append(span(SV_X - SV_L / 2 - 0.4, SV_X + SV_L / 2 + 0.4, Y_D - 1, YS + 1,
                    SV_Z - SV_WID / 2 - 0.4, SV_Z + SV_WID / 2 + 0.4))
    for s in (-1, 1):
        cut.append(cyl_y_span(2.0, Y_D - 1, YS - 1.0, SV_X + s * 13.9, SV_Z))

    # ---- ESP32 zone
    for s in (-1, 1):  # HY-SRF05 rails + PCB grooves
        x0, x1 = sorted((HY_CX + s * 21.8, HY_CX + s * 26.0))
        x0 = max(x0, -XI - 0.5)
        add.append(span(x0, x1, YF - 0.5, HY_PCB_Y + 3.6, FLOOR - 0.5, HY_Z0 + HY_BOARD[1] + 1))
        g0, g1 = sorted((HY_CX + s * 21.6, HY_CX + s * (HY_BOARD[0] / 2 + 0.7)))
        cut.append(span(g0, g1, HY_PCB_Y, HY_PCB_Y + 2.2, HY_Z0, HY_Z0 + 40))
    for s in (-1, 1):
        cut.append(cyl_y(HY_HOLE_D, 10, HY_CX + s * HY_SPACING / 2, Y0, HY_CZ))
    slots = union(box(3.2, 10, FAN_GRILLE_D, FAN_CX + k * 5.2, Y0, FAN_CZ - FAN_GRILLE_D / 2) for k in range(-3, 4))
    cut.append(slots ^ cyl_y(FAN_GRILLE_D, 10, FAN_CX, Y0, FAN_CZ))
    for sx in (-1, 1):
        for sz in (-1, 1):
            cut.append(cyl_y(3.4, 10, FAN_CX + sx * FAN_SCREW / 2, Y0, FAN_CZ + sz * FAN_SCREW / 2))
    for s in (-1, 1):  # ESP32-S3 cradle (along X)
        add.append(box(40.0, 2.0, 6.0, ESP_CX, ESP_CY + s * (ESP_W / 2 + 1.0), FLOOR))
    add.append(box(2.0, 16.0, 6.0, ESP_X0 + ESP_L + 1.0, ESP_CY, FLOOR))
    cut.append(span(-W, -XI + 1, ESP_CY - 11.0, ESP_CY + 11.0, 9.0, 18.0))       # 2x USB-C (left wall)
    for k in range(-2, 3):                                                       # KY-015 vents (left)
        cut.append(box(10, 2.0, 12.0, -W / 2, DHT_CY + k * 3.6, DHT_CZ - 6.0))
    for s in (-1, 1):                                                            # KY-015 guides
        y0, y1 = sorted((DHT_CY + s * 9.5, DHT_CY + s * 11.3))
        add.append(span(-XI - 0.5, -XI + 8.0, y0, y1, FLOOR - 0.5, DHT_CZ + 10))
    for k in range(-2, 3):                                                       # fan intake vents (left, top)
        cut.append(box(10, 2.0, 12.0, -W / 2, -85.0 + k * 4.5, 56.0))
    add.append(pocket(*BB, 3.0, *BB_C))
    cut.append(rbox(10.0, 9.6, 4.0, 1.8, -W / 2, V5_Y, V5_Z - 2.0).rotate((0, 0, 0)))  # 5V IN (USB-C)
    for s in (-1, 1):                                                            # 5V breakout guides
        add.append(span(-XI - 0.5, -XI + 4.0, V5_Y + s * 7.5 - 0.9, V5_Y + s * 7.5 + 0.9, FLOOR - 0.5, V5_Z + 8))

    # ---- Pi zone
    rim = span(PI_CASE_X0 - 1.8, PI_CASE_X1 + 1.8, PI_CASE_Y0 - 1.0, PI_CASE_Y1 + 1.8, FLOOR - 0.5, FLOOR + 3.0)
    rim = rim - span(PI_CASE_X0 - 0.3, PI_CASE_X1 + 0.3, PI_CASE_Y0 - 2.0, PI_CASE_Y1 + 0.3, FLOOR - 1, FLOOR + 4)
    rim = rim - span(PI_CASE_X1 - 1, PI_CASE_X1 + 3, PI_CASE_Y0, PI_CASE_Y1, FLOOR - 1, FLOOR + 4)   # cable side open
    add.append(rim)                                                          # locating rim for the Pi case
    cut.append(span(PI_CASE_X0 + 3, PI_CASE_X1 - 3, Y0 - 1, YF + 1, FLOOR + 3.0, FLOOR + 28.0))  # USB/Eth
    cut.append(rbox(10, 13.0, 9.0, 2.0, W / 2, PI_USBC_Y, PI_PCB_Z + 0.5))                       # Pi POWER
    cut.append(span(XI - 1, W, PI_HDMI_Y[1] - 6.0, PI_HDMI_Y[0] + 6.0, PI_PCB_Z + 0.5, PI_PCB_Z + 9.0))  # HDMI
    for k in range(5):                                                           # right wall vents
        cut.append(box(10, 2.0, 12.0, W / 2, -90.0 + k * 4.5, 52.0))
    return m + union(add) - union(cut) + shelf_ledges()


def shelf_ledges() -> Manifold:
    """45-degree ledges (no support needed) on the guard rib and back wall; the phone floor plate rests on them."""
    L, zt = SHELF_LEDGE, Z_PHONE_FLOOR
    out = []
    for yw, s in ((YP, 1), (YB, -1)):                      # wall face, direction into the bay
        y_far = yw + s * L
        top = span(-XI - 0.5, XI + 0.5, *sorted((yw - s * 0.5, y_far)), zt - 0.6, zt)
        foot = span(-XI - 0.5, XI + 0.5, *sorted((yw - s * 0.5, yw)), zt - L - 0.6, zt)
        out.append((top + foot).hull())
    return union(out)


def phone_floor() -> Manifold:
    """Separate floor plate of the phone compartment: drop in from the top onto the ledges."""
    f = SHELF_FIT
    plate = span(-XI + f, XI - f, YP + f, YB - f, Z_PHONE_FLOOR, Z_PHONE)
    for sx in (-1, 1):
        plate = plate - cyl_z(SHELF_HOLE_D, 10, sx * (XI - 14.0), (YP + YB) / 2, Z_PHONE_FLOOR - 1)
    return plate


# ============================================================== LIDS
def main_lid() -> Manifold:
    plate = rbox(W, D, LID_T, CORNER_R, z0=H) ^ span(-W, W, -D, Y_D + DIV_T / 2, H - 1, H + LID_T + 1)
    lw0, lw1 = -XI + FIT, XI - FIT
    ly0, ly1 = YF + FIT, Y_D - FIT
    lip = span(lw0, lw1, ly0, ly1, H - LIP_H, H) - span(lw0 + LIP_T, lw1 - LIP_T, ly0 + LIP_T, ly1 - LIP_T,
                                                        H - LIP_H - 1, H + 1)
    top = plate + lip + union([
        frame(*MQ_BOARD, 2.5, 1.2, *MQ_C, H),
        frame(*LDR_BOARD, 2.5, 1.2, *LDR_C, H),
        cyl_z(BUZ_D + 3.0, 5.0, *BUZ_C, H - 5.0) - cyl_z(BUZ_D, 7.0, *BUZ_C, H - 6.0),
        # light pipe down to the ESP32-S3 RGB LED
        cyl_z(9.0, H - (ESP_PCB_Z + 1.6 + 4.0), *ESP_LED, ESP_PCB_Z + 1.6 + 4.0),
    ])
    cx, cy = CAM_C
    cut = [
        cyl_z(MQ_HOLE_D, 20, *MQ_C, H - 10),
        cyl_z(6.0, 20, *LDR_C, H - 10),
        cyl_z(5.6, H, *ESP_LED, ESP_PCB_Z),                                  # light pipe bore
        span(X_DIV - 0.6, X_DIV + 0.6, -D, Y_D, H + LID_T - 0.6, H + 5),      # zone line
        box(CAM_BASE[0] + 0.4, CAM_BASE[1] + 0.4, 1.0, cx, cy, H + LID_T - 0.6),
        box(CAM_RIBBON[0], CAM_RIBBON[1], 20, cx, cy + CAM_RIBBON[2], H - 10),
    ]
    for fx, fy in CAM_FIX:
        cut.append(cyl_z(3.4, 20, cx + fx, cy + fy, H - 10))
    for k in range(8):
        a = np.deg2rad(k * 45 + 22.5)
        cut.append(cyl_z(2.4, 20, MQ_C[0] + 14.5 * np.cos(a), MQ_C[1] + 14.5 * np.sin(a), H - 10))
    cut.append(cyl_z(2.0, 20, *BUZ_C, H - 10))
    for k in range(6):
        a = np.deg2rad(k * 60)
        cut.append(cyl_z(2.0, 20, BUZ_C[0] + 3.6 * np.cos(a), BUZ_C[1] + 3.6 * np.sin(a), H - 10))
    for k in range(6):                                                           # Pi heat vents
        cut.append(box(2.0, 20.0, 10.0, 34.0 + k * 4.5, -42.0, H - 5))
    return top - union(cut)


def phone_lid() -> Manifold:
    """Sliding lid: rides in the grooves, slides out through the right wall."""
    y0, y1 = YS - GROOVE_D + 0.3, YB + GROOVE_D - 0.3
    lid = span(-XI + 0.2, W / 2, y0, y1, PL0, PL0 + PL_T)
    lid = lid - cyl_z(18.0, 10, W / 2, (YP + YB) / 2, PL0 - 1)                    # finger notch
    lid = lid - span(SV_SHAFT_X - 10.0, SV_SHAFT_X + 10.0, HORN_Y - 3.5, HORN_Y + 3.5,
                     PL0 - 1, PL0 + PL_T + 1)                                    # servo lock slot
    return lid


# ============================================================== CAMERA MOUNT
def cam_place(m: Manifold) -> Manifold:
    return m.rotate((-CAM_TILT, 0, 0)).translate((CAM_C[0], CAM_C[1] - CAM_BASE[1] / 2, H + LID_T - 0.6))


def camera_mount() -> Manifold:
    bx, by = CAM_BASE
    px, pt, ph = CAM_PLATE
    z0 = H + LID_T - 0.6
    cx, cy = CAM_C
    outer = (box(bx, by, 2.0, cx, cy, z0) + cam_place(box(px, pt, ph, 0, pt / 2, 0))).hull()
    inner = (box(bx - 4, by - 4, 0.01, cx, cy + 0.5, z0 + 2.0)
             + cam_place(box(px - 4, 0.01, ph - 4, 0, pt + 0.01, 2.0))).hull()
    m = outer - inner + union(cyl_z(6.0, 6.0, cx + fx, cy + fy, z0) for fx, fy in CAM_FIX)
    lz = CAM_HOLE_Z + CAM_HOLE_DZ / 2
    cut = [cam_place(rbox(CAM_LENS[0], CAM_LENS[1], 10, 2.0).rotate((90, 0, 0)).translate((0, 5, lz)))]
    for sx in (-1, 1):
        for dz in (0.0, CAM_HOLE_DZ):
            cut.append(cam_place(cyl_y(2.2, 10, sx * CAM_HOLE_DX, 0, CAM_HOLE_Z + dz)))
    for fx, fy in CAM_FIX:
        cut.append(cyl_z(2.6, 5.0, cx + fx, cy + fy, z0 - 0.1))
    cut.append(box(CAM_RIBBON[0], CAM_RIBBON[1], 5, cx, cy + CAM_RIBBON[2], z0 - 1))
    return m - union(cut)


# ============================================================== MOCK-UP COMPONENTS
PCB_GREEN, PCB_BLUE, PCB_PURPLE = "#1f7a4d", "#1d4fa3", "#6a3fa6"
BLACK, SILVER, WHITE, DARK = "#262a30", "#c3c8ce", "#f4f4f2", "#3b3f46"

# key -> (title, description, zone, mounted on main lid?)
PART_INFO = {
    "esp32":  ("ESP32-S3 DevKitC", "Sensor Node: อ่านเซนเซอร์ ส่ง MQTT ทุก 2 วินาที · USB-C 2 ช่องออกผนังซ้าย · ไฟเลี้ยงจาก Power bank #1", "ESP32", False),
    "rgbled": ("RGB LED บนบอร์ด ESP32 (GPIO48/38)", "ไฟสถานะ เขียว/เหลือง/แดง · มองเห็นจากบนฝาผ่านท่อนำแสง (light pipe)", "ESP32", False),
    "hy":     ("HY-SRF05 Ultrasonic", "วัดระยะ/ตรวจว่ามีคนนั่ง · TRIG GPIO7, ECHO GPIO15 ผ่านตัวแบ่ง 1k/2k · ไฟ 5V", "ESP32", False),
    "dht":    ("KY-015 (DHT11)", "อุณหภูมิ/ความชื้น · S → GPIO6 · ไฟ 3V3 · หันเซนเซอร์เข้าช่องระบายผนังซ้าย", "ESP32", False),
    "ldr":    ("KY-018 (LDR วัดแสง)", "วัดแสงบนโต๊ะ · S → GPIO5 (ADC1) · ตัว LDR โผล่ผ่านรู Ø6 บนฝา", "ESP32", True),
    "mq135":  ("MQ135", "คุณภาพอากาศ · AO → GPIO4 (ADC1) · ไฟ 3V3 · ตัวเซนเซอร์โผล่นอกฝา (warm-up 3-5 นาที)", "ESP32", True),
    "fan":    ("พัดลม DC 5V 40 mm", "ESP32 GPIO10 → MOSFET (logic-level) · 60% = WARNING, 100% = NOT OK · ดูดลมจากช่องผนังซ้าย", "ESP32", False),
    "bb":     ("Mini breadboard: 5V rail + MOSFET", "ราง 5V สำหรับ servo/พัดลม/HY-SRF05 + MOSFET IRLZ44N + ไดโอด 1N4007 + C 470µF + ตัวแบ่งแรงดัน", "ESP32", False),
    "v5in":   ("ช่องรับไฟ 5V (USB-C breakout)", "จ่ายไฟให้ servo + พัดลม แยกจาก ESP32 · เสียบสายจาก Power bank #1 (ช่องที่ 2)", "ESP32", False),
    "servo":  ("Servo SG90 (ล็อกฝามือถือ)", "ยึดกับผนังฝั่ง ESP32 แกนลอดผนังเข้าช่องสวิง 15 มม. · แขนตั้ง = ล็อก, แขนนอน = ปลดล็อก · ส้ม → GPIO9, แดง → 5V rail", "ESP32", False),
    "pi":     ("Raspberry Pi 5 (ในกล่อง 9 × 7 ซม.)", "Gateway: Mosquitto + monitor.py ตัดสินใจ OK/WARNING/NOT OK · วางในขอบล็อกบนพื้น · USB/LAN หน้า · USB-C/HDMI ด้านขวา", "Raspberry Pi", False),
    "camera": ("Pi Camera (CSI 22-pin)", "ถ่ายภาพเมื่อ NOT OK · เงยขึ้น 25° · สายแพลงร่องบนฝาไปเสียบ Pi 5", "Raspberry Pi", True),
    "cam_mount": ("ฐานกล้อง (พิมพ์)", "วางในโซนกล้องบนฝา ยึด M3 ×2 จากใต้ฝา · กล้องยึด M2 ×4", "Raspberry Pi", True),
    "buzzer": ("KY-006 Buzzer", "ร้องเตือนเมื่อ NOT OK · S → GPIO18 (ขา 12), - → GND", "Raspberry Pi", True),
    "pb1":    ("Power bank (1 ช่อง)", "เลี้ยง ESP32-S3 + ราง 5V servo/พัดลม เท่านั้น · สายร้อยผ่านรูซ้าย → ช่องร้อยสาย → เข้าโซน ESP32", "Power", False),
    "cable_esp": ("สายไฟ Power bank → ESP32-S3", "ออกปลายซ้ายของ power bank → รูผนังกันมือถือ (ซ้ายล่าง) → ช่องร้อยสาย → รูผนังกั้นฝั่ง ESP32 → USB-C ของ ESP32 (มีที่ว่าง 15 มม. สำหรับหัวปลั๊ก)", "Power", False),
    "cable_pi":  ("สายไฟ Raspberry Pi 5 (ออกนอกกล่อง)", "สายจากกล่อง Pi ลอดรูไฟ Pi ที่ผนังขวา → ไปเสียบปลั๊ก/adapter ข้างนอก (ไม่ใช้ power bank)", "Raspberry Pi", False),
    "phone":  ("มือถือ", "วางในช่องมือถือ เลื่อนฝาปิดจากด้านขวา", "Phone", False),
}


def cable(points, r=2.0) -> Manifold:
    """A round cable through a list of points (hull of spheres per segment)."""
    balls = [Manifold.sphere(r, 16).translate(pt) for pt in points]
    return union((balls[i] + balls[i + 1]).hull() for i in range(len(balls) - 1))


def components():
    c = []
    zt = H - 1.6
    # ESP32-S3 (along X, USB-C at the left wall)
    x0, x1 = ESP_X0, ESP_X0 + 69.0
    c.append(("esp32", span(x0, x1, ESP_CY - 12.75, ESP_CY + 12.75, ESP_PCB_Z, ESP_PCB_Z + 1.6), BLACK))
    for s in (-1, 1):
        c.append(("esp32", span(x0 + 8, x1 - 8, ESP_CY + s * 11.4 - 1.25, ESP_CY + s * 11.4 + 1.25, FLOOR, ESP_PCB_Z), DARK))
    c.append(("esp32", span(x1 - 25.5, x1, ESP_CY - 9, ESP_CY + 9, ESP_PCB_Z + 1.6, ESP_PCB_Z + 4.8), SILVER))
    for dy in (-6.0, 6.0):
        c.append(("esp32", span(x0 - 1, x0 + 7, ESP_CY + dy - 4.5, ESP_CY + dy + 4.5, ESP_PCB_Z + 1.6, ESP_PCB_Z + 4.8), SILVER))
    c.append(("rgbled", box(2.5, 2.5, 1.0, *ESP_LED, ESP_PCB_Z + 1.6), WHITE))
    # HY-SRF05
    c.append(("hy", box(HY_BOARD[0], 1.6, HY_BOARD[1], HY_CX, HY_PCB_Y + 0.8, HY_Z0), PCB_BLUE))
    for s in (-1, 1):
        x = HY_CX + s * HY_SPACING / 2
        c.append(("hy", cyl_y_span(16.0, HY_PCB_Y, HY_PCB_Y - 11.8, x, HY_CZ), SILVER))
        c.append(("hy", cyl_y_span(12.0, HY_PCB_Y - 11.8, HY_PCB_Y - 12.0, x, HY_CZ), BLACK))
    # KY-015
    c.append(("dht", span(-XI, -XI + 5.5, DHT_CY - 7.75, DHT_CY + 7.75, DHT_CZ - 6, DHT_CZ + 6), "#3b8ede"))
    c.append(("dht", span(-XI + 5.5, -XI + 7.1, DHT_CY - 9.25, DHT_CY + 9.25, DHT_CZ - 7.5, DHT_CZ + 7.5), BLACK))
    # KY-018
    c.append(("ldr", box(18.5, 15.0, 1.6, *LDR_C, zt), BLACK))
    c.append(("ldr", cyl_z(5.0, 3.0, *LDR_C, H - 0.5), "#c8743b"))
    # MQ135
    c.append(("mq135", box(32.0, 22.0, 1.6, *MQ_C, zt), PCB_BLUE))
    c.append(("mq135", cyl_z(17.0, 4.0, *MQ_C, H), "#d6c49a"))
    c.append(("mq135", cyl_z(19.0, 11.0, *MQ_C, H + 4.0), SILVER))
    # fan
    fy = YF + FAN_T / 2
    c.append(("fan", box(FAN, FAN_T, FAN, FAN_CX, fy, FAN_CZ - FAN / 2) - cyl_y(38.0, 20, FAN_CX, fy, FAN_CZ), BLACK))
    c.append(("fan", cyl_y(15.0, FAN_T - 1.5, FAN_CX, fy, FAN_CZ), DARK))
    for k in range(7):
        blade = box(17.0, 1.2, 7.0, 15.0, 0, -3.5).rotate((25, 0, 0)).rotate((0, k * 360 / 7, 0))
        c.append(("fan", blade.translate((FAN_CX, fy, FAN_CZ)), DARK))
    # breadboard + MOSFET + capacitor
    c.append(("bb", box(BB[0] - 0.4, BB[1] - 0.4, 9.5, *BB_C, FLOOR), WHITE))
    c.append(("bb", box(10.0, 4.5, 15.0, BB_C[0] - 10, BB_C[1], FLOOR + 9.5), BLACK))
    c.append(("bb", cyl_z(8.0, 13.0, BB_C[0] + 10, BB_C[1] + 5, FLOOR + 9.5), "#2a2f6b"))
    # 5V input breakout
    c.append(("v5in", span(-XI + 0.2, -XI + 1.8, V5_Y - 6.5, V5_Y + 6.5, V5_Z - 6, V5_Z + 8), PCB_PURPLE))
    c.append(("v5in", span(-XI - 1.0, -XI + 6.0, V5_Y - 4.5, V5_Y + 4.5, V5_Z - 1.6, V5_Z + 1.6), SILVER))
    # servo (tabs on the divider, shaft through it, arm up = locked)
    sb = ("servo", "#2b6fd6")
    c.append((sb[0], span(SV_X - SV_L / 2, SV_X + SV_L / 2, SV_BOTTOM_Y, SV_TOP_Y, SV_Z - SV_WID / 2, SV_Z + SV_WID / 2), sb[1]))
    c.append((sb[0], span(SV_X - SV_TAB_L / 2, SV_X + SV_TAB_L / 2, Y_D - SV_TAB_T, Y_D, SV_Z - SV_WID / 2, SV_Z + SV_WID / 2), sb[1]))
    c.append((sb[0], cyl_y_span(11.8, SV_TOP_Y, SV_TOP_Y + SV_HUMP, SV_SHAFT_X, SV_Z), sb[1]))
    c.append(("servo", cyl_y_span(7.0, SV_TOP_Y + SV_HUMP, HORN_Y + ARM_T / 2, SV_SHAFT_X, SV_Z), WHITE))
    c.append(("servo", span(SV_SHAFT_X - ARM_W / 2, SV_SHAFT_X + ARM_W / 2, HORN_Y - ARM_T / 2, HORN_Y + ARM_T / 2,
                            SV_Z - 3, SV_Z + ARM), WHITE))
    # Raspberry Pi 5 in its case (9 x 7 cm)
    cx0, cx1, cy0, cy1 = PI_CASE_X0, PI_CASE_X1, PI_CASE_Y0, PI_CASE_Y1
    case = rbox(PI_CASE[0], PI_CASE[1], PI_CASE[2], 5.0, (cx0 + cx1) / 2, (cy0 + cy1) / 2, FLOOR)
    case = case - span(cx0 + 4, cx1 - 4, cy0 - 1, cy0 + 2, FLOOR + 3, FLOOR + 24)
    c.append(("pi", case, "#e6e7eb"))
    c.append(("pi", cyl_z(26.0, 0.6, (cx0 + cx1) / 2, (cy0 + cy1) / 2 + 12, FLOOR + PI_CASE[2]), DARK))
    t = PI_PCB_Z + 1.6
    for xc in (PI_X0 + 9.0, PI_X0 + 27.0):
        c.append(("pi", span(xc - 7.5, xc + 7.5, cy0 + 0.5, cy0 + 17.0, t, t + 16.0), SILVER))
    c.append(("pi", span(PI_X0 + 37.5, PI_X0 + 54.0, cy0 + 0.5, cy0 + 21.0, t, t + 13.5), SILVER))
    # camera on the mount
    pt = CAM_PLATE[1]
    lz = CAM_HOLE_Z + CAM_HOLE_DZ / 2
    c.append(("camera", cam_place(box(25.0, 1.0, 24.0, 0, pt + 0.5, lz - 9.5)), PCB_GREEN))
    c.append(("camera", cam_place(box(8.5, 4.0, 8.5, 0, pt - 1.6, lz - 4.25)), BLACK))
    c.append(("camera", cam_place(cyl_y(6.0, 1.2, 0, 0.0, lz)), "#0b2a4a"))
    # KY-006
    c.append(("buzzer", cyl_z(12.0, 8.0, *BUZ_C, H - 8.0), BLACK))
    c.append(("buzzer", box(18.5, 15.0, 1.6, *BUZ_C, H - 9.6), BLACK))
    # power banks + phone
    pbx, pby, pbz = POWERBANK
    ycen = (YP + YB) / 2
    for key, z0, col, sx in (("pb1", FLOOR + 1.0, "#4a5160", -1),):
        c.append((key, rbox(pbx, pby, pbz, 6.0, 0, ycen, z0), col))
        xe0, xe1 = sorted((sx * (pbx / 2 - 1), sx * (pbx / 2 + 0.5)))
        c.append((key, span(xe0, xe1, ycen - 15, ycen - 3, z0 + 5, z0 + 11), SILVER))
        c.append((key, span(xe0, xe1, ycen + 3, ycen + 12, z0 + 6, z0 + 9.5), SILVER))
    # power cables (bay -> rib hole -> channel under the swing strip -> divider hole -> board)
    ym = (YS + Y_RIB) / 2
    c.append(("cable_esp", cable([(-pbx / 2 - 4, ycen - 9, FLOOR + 9), (-XI + 8, ycen - 9, FLOOR + 9),
                                  (-XI + 12, YP + 4, 10), (-XI + 12, ym, 10), (-XI + 9, Y_D - 4, 8),
                                  (-XI + 7, ESP_CY - 6, 6), (ESP_X0 - 9, ESP_CY - 6, ESP_PCB_Z + 3)]), "#f08a24"))
    c.append(("cable_esp", span(ESP_X0 - 11, ESP_X0 - 1, ESP_CY - 10.5, ESP_CY - 1.5, ESP_PCB_Z + 0.5, ESP_PCB_Z + 6), DARK))
    zc = PI_PCB_Z + 6
    c.append(("cable_pi", cable([(PI_CASE_X1 + 8, PI_USBC_Y, zc), (W / 2 + 8, PI_USBC_Y, zc),
                                 (W / 2 + 14, PI_USBC_Y - 4, 3), (W / 2 + 14, PI_USBC_Y - 40, 2)]), "#e5322d"))
    c.append(("cable_pi", span(PI_X1 + 1.5, PI_CASE_X1 + 10, PI_USBC_Y - 4.5, PI_USBC_Y + 4.5, PI_PCB_Z + 1, PI_PCB_Z + 11), DARK))
    c.append(("phone", rbox(160.0, 75.0, 8.5, 9.0, 0, ycen, Z_PHONE + 0.5), "#30343b"))
    c.append(("phone", rbox(150.0, 69.0, 0.3, 6.0, 0, ycen, Z_PHONE + 9.0), "#0d1b2a"))
    return c


def swing_report() -> dict:
    """Clearances around the servo arm (mm)."""
    horn_y0, horn_y1 = HORN_Y - ARM_T / 2, HORN_Y + ARM_T / 2
    return {
        "arm plane -> divider (front)": horn_y0 - YS,
        "arm plane -> guard rib (back)": Y_RIB - horn_y1,
        "arm tip (pointing down) -> strip floor": (SV_Z - ARM) - (STRIP_SHELF + 2),
        "arm tip (sideways) -> side walls": min(SV_SHAFT_X - ARM - (-XI), XI - (SV_SHAFT_X + ARM)),
        "strip width (Y)": SWING,
        "arm tip into lid slot": TIP_IN,
    }


# ============================================================== EXPORT / RENDER
def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=False)


def moved(tm, dx=0.0, dy=0.0, dz=0.0):
    t = tm.copy()
    t.apply_translation((dx, dy, dz))
    return t


def render(meshes_colors, path, title, elev=24, azim=-58, size=(1200, 900), ss=2):
    """Small numpy z-buffer rasterizer + depth-edge outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    az, el = np.deg2rad(azim), np.deg2rad(elev)
    e = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.array([-np.sin(az), np.cos(az), 0.0])
    u = np.cross(e, r)
    light = 0.55 * e + 0.45 * np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)

    V, F, C, N = [], [], [], []
    off = 0
    for tm, color in meshes_colors:
        V.append(tm.vertices); F.append(tm.faces + off); off += len(tm.vertices)
        C.append(np.tile(matplotlib.colors.to_rgb(color), (len(tm.faces), 1)))
        N.append(tm.face_normals)
    V, F, C, N = np.vstack(V), np.vstack(F), np.vstack(C), np.vstack(N)
    P = np.stack([V @ r, V @ u, V @ e], 1)
    W_, H_ = size[0] * ss, size[1] * ss
    mn, mx = P[:, :2].min(0), P[:, :2].max(0)
    scale = 0.88 * min(W_ / (mx[0] - mn[0]), (H_ * 0.82) / (mx[1] - mn[1]))
    cen = (mn + mx) / 2
    sx = (P[:, 0] - cen[0]) * scale + W_ / 2
    sy = H_ * 0.555 - (P[:, 1] - cen[1]) * scale
    z = P[:, 2]

    zbuf = np.full((H_, W_), -np.inf)
    img = np.ones((H_, W_, 3))
    shade = np.abs(N @ light) * 0.7 + 0.3
    fcol = np.clip(C * shade[:, None], 0, 1)
    for i, (a, b, c) in enumerate(F):
        x0 = int(max(min(sx[a], sx[b], sx[c]), 0)); x1 = int(min(max(sx[a], sx[b], sx[c]) + 1, W_ - 1))
        y0 = int(max(min(sy[a], sy[b], sy[c]), 0)); y1 = int(min(max(sy[a], sy[b], sy[c]) + 1, H_ - 1))
        if x1 < x0 or y1 < y0:
            continue
        d = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c])
        if abs(d) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((sy[b] - sy[c]) * (xs - sx[c]) + (sx[c] - sx[b]) * (ys - sy[c])) / d
        w1 = ((sy[c] - sy[a]) * (xs - sx[c]) + (sx[a] - sx[c]) * (ys - sy[c])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = m & (zz > sub)
        sub[upd] = zz[upd]
        img[y0:y1 + 1, x0:x1 + 1][upd] = fcol[i]

    zf = np.where(np.isfinite(zbuf), zbuf, -1e4)
    gx = np.abs(np.diff(zf, axis=1, append=zf[:, -1:]))
    gy = np.abs(np.diff(zf, axis=0, append=zf[-1:, :]))
    edge = np.maximum(gx, gy) > 1.2
    img[edge] *= 0.25
    img = img.reshape(size[1], ss, size[0], ss, 3).mean((1, 3))

    fig = plt.figure(figsize=(size[0] / 140, size[1] / 140), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    ax.text(0.03, 0.965, title, transform=ax.transAxes, va="top", fontsize=11, color="#1d2330")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Desk Box</title>
<style>
:root{--bg:#f4f5f7;--panel:#ffffff;--ink:#1d2330;--muted:#5d6575;--accent:#2f6fde;--line:#dde1e8;--tip:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--panel:#1d2128;--ink:#e8ebf0;--muted:#9aa3b2;--accent:#6ea0ff;--line:#2c323c;--tip:#232831}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Segoe UI",sans-serif}
#app{display:grid;grid-template-columns:310px 1fr;height:100%}
aside{background:var(--panel);border-right:1px solid var(--line);padding:16px;overflow:auto}
h1{font-size:16px;margin:0 0 4px}
p.sub{margin:0 0 12px;color:var(--muted);font-size:12px}
label{display:flex;gap:8px;align-items:center;margin:7px 0;font-size:13px}
.presets{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:6px 0 4px}
.presets button{font:inherit;font-size:12px;padding:6px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.presets button:hover{border-color:var(--accent)}
input[type=range]{width:100%}
h2{font-size:12px;margin:14px 0 4px;padding-top:10px;border-top:1px solid var(--line);display:flex;align-items:center;gap:6px}
h2 .dot{width:8px;height:8px;border-radius:50%}
ul{margin:0;padding:0;list-style:none;font-size:12.5px}
li{display:flex;gap:8px;align-items:center;padding:3px 6px;border-radius:6px;cursor:default}
li:hover,li.hl{background:color-mix(in srgb,var(--accent) 14%,transparent)}
.sw{width:10px;height:10px;border-radius:2px;flex:none;border:1px solid var(--line)}
#view{position:relative;overflow:hidden}
canvas{display:block}
#tip{position:absolute;pointer-events:none;max-width:290px;background:var(--tip);color:var(--ink);border:1px solid var(--line);
 border-radius:10px;padding:8px 10px;font-size:12.5px;box-shadow:0 6px 24px rgba(0,0,0,.18);display:none}
#tip b{display:block;font-size:13.5px;margin-bottom:2px}
#tip .zone{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;color:#fff;margin-bottom:5px}
@media (max-width:700px){#app{grid-template-columns:1fr;grid-template-rows:auto 1fr}aside{border-right:0;border-bottom:1px solid var(--line);max-height:42vh}}
</style></head><body>
<div id="app">
<aside>
 <h1>Smart Study Desk – กล่องเดียว</h1>
 <p class="sub">เอาเมาส์ชี้ที่ชิ้นไหนก็ได้เพื่อดูว่าคืออะไร · ลากเพื่อหมุน · scroll เพื่อซูม</p>
 <h2>แบ่งชั้น (Layers)</h2>
 <div class="presets">
  <button data-p="all">ทั้งหมด</button><button data-p="top">ชั้นบน: ฝา</button>
  <button data-p="mid">ชั้นอุปกรณ์</button><button data-p="low">ชั้นล่าง: power bank</button>
 </div>
 <label><input type="checkbox" class="layer" id="L_lid" checked> ฝาบน + ฐานกล้อง</label>
 <label><input type="checkbox" class="layer" id="L_plid" checked> ฝาช่องมือถือ</label>
 <label><input type="checkbox" class="layer" id="L_body" checked> ตัวกล่อง</label>
 <label><input type="checkbox" class="layer" id="L_esp" checked> อุปกรณ์โซน ESP32-S3</label>
 <label><input type="checkbox" class="layer" id="L_pi" checked> อุปกรณ์โซน Raspberry Pi</label>
 <label><input type="checkbox" class="layer" id="L_power" checked> Power bank + สายไฟ</label>
 <label><input type="checkbox" class="layer" id="L_phone" checked> มือถือ</label>
 <label>ตัดดูตามความสูง: เห็นถึง z = <b id="zv">สูงสุด</b></label>
 <input type="range" id="clipz" min="0" max="80" value="80">
 <label>ตัดดูจากด้านหน้า: ตัดออก <b id="yv">0</b> มม.</label>
 <input type="range" id="clipy" min="0" max="205" value="0">
 <h2>อื่น ๆ</h2>
 <label><input type="checkbox" id="ghost"> กล่องโปร่งใส (มองทะลุ)</label>
 <label>ยกฝาอิเล็กทรอนิกส์</label>
 <input type="range" id="lift" min="0" max="60" value="35">
 <label>เลื่อนฝามือถือออก (ไปทางขวา)</label>
 <input type="range" id="slide" min="0" max="170" value="0">
 <label><input type="checkbox" id="unlock"> servo ปลดล็อก (แขนนอน)</label>
 <div id="lists"></div>
</aside>
<div id="view"><div id="tip"></div></div>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
const DATA = __DATA__;
const ZC = {"ESP32":"#1f9d74","Raspberry Pi":"#c2185b","Phone":"#7c4dff","Power":"#e08a00"};
const view = document.getElementById('view'), tip = document.getElementById('tip');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio);
renderer.localClippingEnabled = true;
const clipZ = new THREE.Plane(new THREE.Vector3(0,0,-1), 1000), clipY = new THREE.Plane(new THREE.Vector3(0,1,0), 1000);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark?0x14171c:0xf4f5f7);
const cam = new THREE.PerspectiveCamera(35,1,1,5000); cam.up.set(0,0,1);
const ctl = new OrbitControls(cam, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x667788,1.1));
const dl = new THREE.DirectionalLight(0xffffff,1.5); dl.position.set(120,-200,260); scene.add(dl);
const grid = new THREE.GridHelper(600,60,dark?0x3a414d:0xc9ced8,dark?0x262b33:0xe3e6ec);
grid.rotation.x = Math.PI/2; scene.add(grid);
const loader = new STLLoader();
const meshes = [];
for (const p of DATA.pieces){
  const bin = Uint8Array.from(atob(p.stl), c=>c.charCodeAt(0)).buffer;
  const geo = loader.parse(bin); geo.computeVertexNormals();
  if (p.pivot){ geo.translate(-p.pivot[0],-p.pivot[1],-p.pivot[2]); }
  const mat = new THREE.MeshStandardMaterial({color:p.color, roughness:.55, metalness:p.kind==='comp'?.15:.03, transparent:true, emissive:0x000000, side:THREE.DoubleSide, clippingPlanes:[clipZ, clipY]});
  const m = new THREE.Mesh(geo, mat); m.userData = p; scene.add(m); meshes.push(m);
}
const lists = document.getElementById('lists'); const liByKey = {};
for (const zone of ["ESP32","Raspberry Pi","Power","Phone"]){
  const h = document.createElement('h2'); h.innerHTML = `<span class="dot" style="background:${ZC[zone]}"></span>${DATA.zoneTitle[zone]}`;
  lists.appendChild(h); const ul = document.createElement('ul'); lists.appendChild(ul);
  for (const [key,info] of Object.entries(DATA.info)) if (info.zone===zone){
    const li = document.createElement('li'); li.innerHTML = `<span class="sw" style="background:${info.color}"></span>${info.title}`;
    li.onmouseenter=()=>highlight(key); li.onmouseleave=()=>highlight(null); ul.appendChild(li); liByKey[key]=li;
  }
}
let cur = null;
function highlight(key){
  if (key===cur) return; cur = key;
  for (const m of meshes) m.material.emissive.setHex(key && m.userData.key===key ? 0xffb300 : 0x000000);
  for (const [k,li] of Object.entries(liByKey)) li.classList.toggle('hl', k===key);
}
const ZG = {"ESP32":"esp","Raspberry Pi":"pi","Power":"power","Phone":"phone"};
function groupOf(p){
  if (p.kind==='lid' || p.kind==='cam') return 'lid';
  if (p.kind==='plid') return 'plid';
  if (p.kind==='body' || p.kind==='pfloor') return 'body';
  return ZG[DATA.info[p.key].zone];
}
const on = id => document.getElementById('L_'+id).checked;
let ZCUT = 1e9, YCUT = -1e9;
function update(){
  const ghost=document.getElementById('ghost').checked;
  const zc = +document.getElementById('clipz').value, yc = +document.getElementById('clipy').value;
  ZCUT = zc >= 80 ? 1e9 : zc; YCUT = yc <= 0 ? -1e9 : DATA.regions.yFront + yc;
  clipZ.constant = ZCUT; clipY.constant = -YCUT;
  document.getElementById('zv').textContent = zc >= 80 ? 'สูงสุด' : zc + ' มม.';
  document.getElementById('yv').textContent = yc;
  const lift=+document.getElementById('lift').value, slide=+document.getElementById('slide').value;
  const unlock=document.getElementById('unlock').checked;
  for (const m of meshes){
    const p=m.userData, shell=(p.kind!=='comp');
    m.visible = on(groupOf(p)) && (!p.onLid || on('lid'));
    m.position.set(0,0,0); m.rotation.set(0,0,0);
    if (p.kind==='lid' || p.onLid) m.position.z = lift;
    if (p.kind==='plid' || p.key==='phone_slide') m.position.x = slide;
    if (p.pivot){ m.position.x += p.pivot[0]; m.position.y += p.pivot[1]; m.position.z += p.pivot[2];
                  if (unlock) m.rotation.y = Math.PI/2; }
    const op = (ghost && shell && p.kind!=='cam') ? 0.18 : 1;
    m.material.opacity = op; m.material.depthWrite = op===1;
    p.pickable = op===1;
  }
}
['ghost','lift','slide','unlock','clipz','clipy'].forEach(id=>document.getElementById(id).addEventListener('input',update));
document.querySelectorAll('.layer').forEach(el=>el.addEventListener('input',update));
const PRESET = {
  all: {L:['lid','plid','body','esp','pi','power','phone'], z:80},
  top: {L:['lid','plid','esp','pi'], z:80},
  mid: {L:['body','esp','pi','phone'], z:62},
  low: {L:['body','power'], z:38},
};
document.querySelectorAll('.presets button').forEach(b=>b.onclick=()=>{
  const pr = PRESET[b.dataset.p];
  for (const g of ['lid','plid','body','esp','pi','power','phone']) document.getElementById('L_'+g).checked = pr.L.includes(g);
  document.getElementById('clipz').value = pr.z; update();
});
function shellInfo(p, pt){
  const R = DATA.regions, s = DATA.shell;
  if (p.kind==='plid') return s.plid;
  if (p.kind==='pfloor') return s.pfloor;
  if (p.kind==='lid') return pt.x < R.xDiv ? s.lid_esp : s.lid_pi;
  if (pt.y > R.yDiv){
    if (pt.y < R.yRib) return pt.z < R.zStrip ? s.channel : s.swing;
    if (pt.z < R.zPhone) return s.bay;
    return s.phone;
  }
  return pt.x < R.xDiv ? s.body_esp : s.body_pi;
}
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener('pointermove', e=>{
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
  ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObjects(meshes.filter(m=>m.visible && m.userData.pickable)).find(h=>h.point.z<=ZCUT+0.01 && h.point.y>=YCUT-0.01);
  if (!hit){ tip.style.display='none'; highlight(null); return; }
  const p = hit.object.userData;
  const info = (p.kind==='comp' || p.kind==='cam') ? DATA.info[p.key] : shellInfo(p, hit.point.clone().sub(hit.object.position));
  highlight(p.key || null);
  tip.innerHTML = `<span class="zone" style="background:${ZC[info.zone]}">${DATA.zoneTitle[info.zone]}</span><b>${info.title}</b>${info.desc}`;
  tip.style.display='block';
  const x = e.clientX-r.left+14, y = e.clientY-r.top+14;
  tip.style.left = Math.min(x, r.width-tip.offsetWidth-8)+'px'; tip.style.top = Math.min(y, r.height-tip.offsetHeight-8)+'px';
});
renderer.domElement.addEventListener('pointerleave', ()=>{tip.style.display='none'; highlight(null);});
update();
const b = new THREE.Box3(); meshes.forEach(o=>b.expandByObject(o));
const c = b.getCenter(new THREE.Vector3()); const s = b.getSize(new THREE.Vector3()).length();
ctl.target.copy(c); cam.position.set(c.x+s*0.55, c.y-s*1.1, c.z+s*0.8); ctl.update();
function resize(){const r=view.getBoundingClientRect();renderer.setSize(r.width,r.height);cam.aspect=r.width/r.height;cam.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop);renderer.render(scene,cam);})();
</script></body></html>
"""

BODY_COLOR, LID_COLOR, PLID_COLOR, CAM_COLOR = "#cdd3dc", "#f2a93b", "#9fb1e8", "#e8734a"
PFLOOR_COLOR = "#7fc8a9"


def b64stl(tm: trimesh.Trimesh) -> str:
    return base64.b64encode(trimesh.exchange.stl.export_stl(tm)).decode()


def main():
    STL_DIR.mkdir(exist_ok=True)
    RENDER_DIR.mkdir(exist_ok=True)

    bd, ml, pl, cm, pf = body(), main_lid(), phone_lid(), camera_mount(), phone_floor()
    report = []
    for name, m, orient in (("smart_desk_box_body", bd, to_bed), ("smart_desk_box_lid", ml, flip_to_bed),
                            ("phone_lid", pl, to_bed), ("camera_mount", cm, to_bed),
                            ("phone_floor", pf, to_bed)):
        assert m.status().name == "NoError", (name, m.status())
        tm = to_trimesh(orient(m))
        tm.export(STL_DIR / f"{name}.stl")
        report.append((name, tm.extents, tm.volume / 1000.0, tm.is_watertight))

    bd_t, ml_t, pl_t, cm_t, pf_t = map(to_trimesh, (bd, ml, pl, cm, pf))
    comps = [(k, to_trimesh(m), col) for k, m, col in components()]
    on_lid = lambda k: PART_INFO[k][3]

    def scene(lift=0.0, slide=0.0, with_comp=True, keep=None):
        out = [(bd_t, BODY_COLOR), (pf_t, PFLOOR_COLOR), (moved(ml_t, dz=lift), LID_COLOR), (moved(cm_t, dz=lift), CAM_COLOR),
               (moved(pl_t, dx=slide), PLID_COLOR)]
        if with_comp:
            out += [(moved(t, dz=lift) if on_lid(k) else t, col) for k, t, col in comps if keep is None or k in keep]
        return out

    render(scene(), RENDER_DIR / "box_closed.png",
           "Smart Study Desk – one merged box\nfront: HY-SRF05 + 5V fan (ESP32-S3)  |  Pi 5 USB/LAN   ·   top: MQ135, KY-018, LED pipe | camera, KY-006")
    render(scene(), RENDER_DIR / "box_right_back.png",
           "Right side: Pi POWER hole (Pi cable out to a wall plug) + HDMI, power bank slot, phone lid exit", elev=22, azim=35)
    render(scene(), RENDER_DIR / "box_left.png",
           "Left side: ESP32-S3 USB-C x2, 5V IN (servo/fan), KY-015 vents, fan intake, power bank slot", elev=22, azim=-150)
    inside = [(bd_t, BODY_COLOR), (pf_t, PFLOOR_COLOR)] + [(t, col) for k, t, col in comps if not on_lid(k) and k not in ("phone",)]
    render(inside, RENDER_DIR / "box_inside.png",
           "Inside (lids removed)\nESP32-S3 zone | Pi 5 zone · servo through the divider · swing strip · power banks", elev=62, azim=-70)
    low = span(-W, W, -D, D, -1, Z_PHONE_FLOOR - 0.5)
    route = [(to_trimesh(bd ^ low), BODY_COLOR)] + [
        (t, col) for k, t, col in comps if k in ("pb1", "esp32", "pi", "cable_esp", "cable_pi", "bb", "fan", "hy", "dht", "v5in")]
    render(route, RENDER_DIR / "cable_route.png",
           "Power (box cut at z = 39.5): 1 power bank -> ESP32-S3 (orange, inside the box)\n"
           "Pi 5 power cable (red) goes out through the Pi power hole to a wall plug", elev=60, azim=-65)
    render([(bd_t, BODY_COLOR), (moved(pf_t, dz=45), PFLOOR_COLOR)], RENDER_DIR / "phone_floor_split.png",
           "Phone floor = separate plate (green): print flat, drop in from the top onto the 45-degree ledges\n"
           "(rib + back wall) -> the body prints with no 86 mm bridge over the power bank bay", elev=30, azim=-150)
    render(scene(slide=150), RENDER_DIR / "phone_open.png",
           "Phone lid slid out to the right: servo arm (white) in the swing strip, phone, power banks below", elev=50, azim=-35)

    # section through the servo shaft, with the 5 mm clearance envelope (red ring)
    cut_keep = span(SV_SHAFT_X - 45, SV_SHAFT_X + 45, HORN_Y - ARM_T / 2 - 0.4, D, -1, H + 20)
    ring = (cyl_y(2 * (ARM + SWING_CLEAR) + 1.2, 0.6, SV_SHAFT_X, HORN_Y - ARM_T / 2 - 0.2, SV_Z)
            - cyl_y(2 * (ARM + SWING_CLEAR) - 1.2, 1.0, SV_SHAFT_X, HORN_Y - ARM_T / 2 - 0.2, SV_Z))
    sec = [(to_trimesh(bd ^ cut_keep), BODY_COLOR), (to_trimesh(pl ^ cut_keep), PLID_COLOR),
           (to_trimesh(ml ^ cut_keep), LID_COLOR), (to_trimesh(ring), "#e5322d")]
    sec += [(to_trimesh(m ^ cut_keep), col) for k, m, col in components() if k in ("servo", "pb1", "phone")]
    render(sec, RENDER_DIR / "section_servo.png",
           "Section at the servo shaft: arm locked (up into the phone-lid slot)\nred ring = arm swing + 5 mm clearance inside the 15 mm strip",
           elev=4, azim=-90)

    # ---------------- viewer data
    pieces = [
        {"kind": "body", "color": BODY_COLOR, "stl": b64stl(bd_t)},
        {"kind": "lid", "color": LID_COLOR, "stl": b64stl(ml_t)},
        {"kind": "plid", "color": PLID_COLOR, "stl": b64stl(pl_t)},
        {"kind": "pfloor", "color": PFLOOR_COLOR, "stl": b64stl(pf_t)},
        {"kind": "cam", "key": "cam_mount", "onLid": True, "color": CAM_COLOR, "stl": b64stl(cm_t)},
    ]
    merged = {}
    arm_parts = []
    for k, m, col in components():
        if k == "servo" and col == WHITE and m.bounding_box()[5] > SV_Z + 5:
            arm_parts.append(to_trimesh(m))
            continue
        merged.setdefault((k, col), []).append(to_trimesh(m))
    for (k, col), tms in merged.items():
        pieces.append({"kind": "comp", "key": k, "onLid": on_lid(k), "color": col,
                       "stl": b64stl(trimesh.util.concatenate(tms))})
    pieces.append({"kind": "comp", "key": "servo", "onLid": False, "color": WHITE,
                   "pivot": [SV_SHAFT_X, HORN_Y, SV_Z], "stl": b64stl(trimesh.util.concatenate(arm_parts))})
    first_color = {"cam_mount": CAM_COLOR}
    for k, _, col in comps:
        first_color.setdefault(k, col)
    info = {k: {"title": v[0], "desc": v[1], "zone": v[2], "color": first_color[k]} for k, v in PART_INFO.items()}
    shell = {
        "body_esp": {"title": "ตัวกล่อง – โซน ESP32-S3", "desc": "รางเสียบ HY-SRF05, ตะแกรงพัดลม, รางวาง ESP32-S3, รางยึด KY-015, ช่อง breadboard, ช่อง 5V IN", "zone": "ESP32"},
        "body_pi": {"title": "ตัวกล่อง – โซน Raspberry Pi 5", "desc": "ขอบล็อกกล่อง Pi 9 × 7 ซม., ช่อง USB/LAN ด้านหน้า, ที่ว่างหัวปลั๊ก 13 มม. + รูไฟ Pi (USB-C) และ HDMI ผนังขวา", "zone": "Raspberry Pi"},
        "swing": {"title": "ช่องสวิง servo (15 มม.)", "desc": "แกน servo ลอดผนังเข้ามาตรงนี้ แขนหมุนโดยมีระยะห่างรอบตัว ≥ 5 มม.", "zone": "ESP32"},
        "channel": {"title": "ช่องร้อยสายไฟ (ใต้ช่องสวิง)", "desc": "สายจาก power bank ลอดรูผนังกันมือถือ (ซ้ายหรือขวา) → วิ่งใต้ช่องสวิง → ลอดรูผนังกั้นเข้าโซน ESP32", "zone": "Power"},
        "bay": {"title": "ช่องใส่ Power bank (1 ช่อง)", "desc": "ช่องเดียว สูง 37 มม. กว้าง 82 มม. ยาวตลอดกล่อง · สอดเข้าได้ทั้งซ้ายและขวา · สายไป ESP32 ลอดรูผนังกันมือถือเข้าช่องร้อยสาย", "zone": "Power"},
        "phone": {"title": "ช่องใส่มือถือ", "desc": "168 × 86 × 24 มม. · มีผนังกันไม่ให้มือถือเข้าไปในช่องสวิง servo", "zone": "Phone"},
        "pfloor": {"title": "พื้นช่องมือถือ (ชิ้นแยก)", "desc": "พิมพ์แยกแบบแบน แล้ววางลงจากด้านบนบนบ่ารับ 45° ที่ผนังกันมือถือ + ผนังหลัง · มีรูนิ้ว 2 รูไว้ยกออกเพื่อเข้าถึงช่อง power bank", "zone": "Phone"},
        "plid": {"title": "ฝาเลื่อนช่องมือถือ", "desc": "เลื่อนออกทางขวา · มีร่องให้แขน servo ล็อก + รอยเว้าไว้ใช้นิ้วดึง", "zone": "Phone"},
        "lid_esp": {"title": "ฝา – โซน ESP32-S3", "desc": "รู MQ135 + รูระบาย, รู KY-018, ท่อนำแสง RGB LED", "zone": "ESP32"},
        "lid_pi": {"title": "ฝา – โซน Raspberry Pi 5", "desc": "โซนวางกล้อง (ร่อง + M3 + ร่องสายแพ), รูเสียง KY-006, ช่องระบายความร้อน Pi", "zone": "Raspberry Pi"},
    }
    data = {"pieces": pieces, "info": info, "shell": shell,
            "regions": {"xDiv": X_DIV, "yDiv": Y_D + DIV_T / 2, "yRib": Y_RIB + RIB_T / 2, "zPhone": Z_PHONE_FLOOR + 1, "zStrip": STRIP_SHELF + 2.5, "yFront": Y0},
            "zoneTitle": {"ESP32": "โซน ESP32-S3 (Sensor Node)", "Raspberry Pi": "โซน Raspberry Pi 5 (Gateway)",
                          "Power": "ช่อง Power bank", "Phone": "ช่องมือถือ"}}
    (HERE / "viewer.html").write_text(VIEWER_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)),
                                      encoding="utf-8")

    print(f"{'file':24s} {'size X x Y x Z (mm)':>22s} {'vol cm3':>8s} watertight")
    for name, ext, vol, wt in report:
        print(f"{name:24s} {ext[0]:6.1f} x {ext[1]:5.1f} x {ext[2]:5.1f} {vol:8.1f} {wt}")
    print("\nservo swing clearances (mm):")
    for k, v in swing_report().items():
        print(f"  {k:40s} {v:6.1f}")


if __name__ == "__main__":
    main()
