#!/usr/bin/python
#
# DS20k labels
# Cityscapes labels
#

from collections import namedtuple

# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# label and all information

Label = namedtuple("Label", ["name", "Id", "trainId", "color"])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

ds20k_labels = [
    #       name               Id   trainId    color
    Label("BACKGROUND", 0, 0, (0, 0, 0)),
    Label("PERSON", 1, 1, (255, 0, 121)),
    Label("CAR", 2, 2, (255, 15, 15)),
    Label("TRUCK", 3, 3, (254, 83, 1)),
    Label("DRIVABLE", 4, 4, (0, 255, 0)),
    Label("NONDRIVABLE", 5, 5, (255, 255, 0)),
    Label("BLOCKER", 6, 6, (192, 192, 192)),
    Label("INFO", 7, 7, (0, 0, 255)),
    Label("SKY", 8, 8, (128, 255, 255)),
    Label("BUILDINGS", 9, 9, (83, 0, 0)),
    Label("NATURE", 10, 10, (0, 80, 0)),
    Label("SLOWAREA", 11, 11, (0, 255, 255)),
    Label("LINES", 12, 12, (128, 0, 255)),
]

cs_labels = [
    #       name                       Id   trainId    color
    Label("unlabeled", 0, 255, (0, 0, 0)),
    Label("ego vehicle", 1, 255, (0, 0, 0)),
    Label("rectification border", 2, 255, (0, 0, 0)),
    Label("out of roi", 3, 255, (0, 0, 0)),
    Label("static", 4, 255, (0, 0, 0)),
    Label("dynamic", 5, 255, (111, 74, 0)),
    Label("ground", 6, 255, (81, 0, 81)),
    Label("road", 7, 0, (128, 64, 128)),
    Label("sidewalk", 8, 1, (244, 35, 232)),
    Label("parking", 9, 255, (250, 170, 160)),
    Label("rail track", 10, 255, (230, 150, 140)),
    Label("building", 11, 2, (70, 70, 70)),
    Label("wall", 12, 3, (102, 102, 156)),
    Label("fence", 13, 4, (190, 153, 153)),
    Label("guard rail", 14, 255, (180, 165, 180)),
    Label("bridge", 15, 255, (150, 100, 100)),
    Label("tunnel", 16, 255, (150, 120, 90)),
    Label("pole", 17, 5, (153, 153, 153)),
    Label("polegroup", 18, 255, (153, 153, 153)),
    Label("traffic light", 19, 6, (250, 170, 30)),
    Label("traffic sign", 20, 7, (220, 220, 0)),
    Label("vegetation", 21, 8, (107, 142, 35)),
    Label("terrain", 22, 9, (152, 251, 152)),
    Label("sky", 23, 10, (70, 130, 180)),
    Label("person", 24, 11, (220, 20, 60)),
    Label("rider", 25, 12, (255, 0, 0)),
    Label("car", 26, 13, (0, 0, 142)),
    Label("truck", 27, 14, (0, 0, 70)),
    Label("bus", 28, 15, (0, 60, 100)),
    Label("caravan", 29, 255, (0, 0, 90)),
    Label("trailer", 30, 255, (0, 0, 110)),
    Label("train", 31, 16, (0, 80, 100)),
    Label("motorcycle", 32, 17, (0, 0, 230)),
    Label("bicycle", 33, 18, (119, 11, 32)),
    Label("license plate", -1, -1, (0, 0, 142)),
]

a2d2_labels = [
    #       name                    id          trainId                color
    Label("Animals", 0, 0, (204, 255, 153)),
    Label("Bicycle 1", 1, 1, (182, 89, 6)),
    Label("Bicycle 2", 2, 2, (150, 50, 4)),
    Label("Bicycle 3", 3, 3, (90, 30, 1)),
    Label("Bicycle 4", 4, 4, (90, 30, 30)),
    Label("Blurred area", 5, 5, (96, 69, 143)),
    Label("Buildings", 6, 6, (241, 230, 255)),
    Label("Car 1", 7, 7, (255, 0, 0)),
    Label("Car 2", 8, 8, (200, 0, 0)),
    Label("Car 3", 9, 9, (150, 0, 0)),
    Label("Car 4", 10, 10, (128, 0, 0)),
    Label("Curbstone", 11, 11, (128, 128, 0)),
    Label("Dashed line", 12, 12, (128, 0, 255)),
    Label("Drivable cobblestone", 13, 13, (180, 50, 180)),
    Label("Ego car", 14, 14, (72, 209, 204)),
    Label("Electronic traffic", 15, 15, (255, 70, 185)),
    Label("Grid structure", 16, 16, (238, 162, 173)),
    Label("Irrelevant signs", 17, 17, (64, 0, 64)),
    Label("Nature object", 18, 18, (147, 253, 194)),
    Label("Non-drivable street", 19, 19, (139, 99, 108)),
    Label("Obstacles / trash", 20, 20, (255, 0, 128)),
    Label("Painted driv. instr.", 21, 21, (200, 125, 210)),
    Label("Parking area", 22, 22, (150, 150, 200)),
    Label("Pedestrian 1", 23, 23, (204, 153, 255)),
    Label("Pedestrian 2", 24, 24, (189, 73, 155)),
    Label("Pedestrian 3", 25, 25, (239, 89, 191)),
    Label("Poles", 26, 26, (255, 246, 143)),
    Label("RD normal street", 27, 27, (255, 0, 255)),
    Label("RD restricted area", 28, 28, (150, 0, 150)),
    Label("Rain dirt", 29, 29, (53, 46, 82)),
    Label("Road blocks", 30, 30, (185, 122, 87)),
    Label("Sidebars", 31, 31, (233, 100, 0)),
    Label("Sidewalk", 32, 32, (180, 150, 200)),
    Label("Signal corpus", 33, 33, (33, 44, 177)),
    Label("Sky", 34, 34, (135, 206, 255)),
    Label("Slow drive area", 35, 35, (238, 233, 191)),
    Label("Small vehicles 1", 36, 36, (0, 255, 0)),
    Label("Small vehicles 2", 37, 37, (0, 200, 0)),
    Label("Small vehicles 3", 38, 38, (0, 150, 0)),
    Label("Solid line", 39, 39, (255, 193, 37)),
    Label("Speed bumper", 40, 40, (110, 110, 0)),
    Label("Tractor", 41, 41, (0, 0, 100)),
    Label("Traffic guide obj.", 42, 42, (159, 121, 238)),
    Label("Traffic sign 1", 43, 43, (0, 255, 255)),
    Label("Traffic sign 2", 44, 44, (30, 220, 220)),
    Label("Traffic sign 3", 45, 45, (60, 157, 199)),
    Label("Traffic signal 1", 46, 46, (0, 128, 255)),
    Label("Traffic signal 2", 47, 47, (30, 28, 158)),
    Label("Traffic signal 3", 48, 48, (60, 28, 100)),
    Label("Truck 1", 49, 49, (255, 128, 0)),
    Label("Truck 2", 50, 50, (200, 128, 0)),
    Label("Truck 3", 51, 51, (150, 128, 0)),
    Label("Utility vehicle 1", 52, 52, (255, 255, 0)),
    Label("Utility vehicle 2", 53, 53, (255, 255, 200)),
    Label("Zebra crossing", 54, 54, (210, 50, 115)),
]
