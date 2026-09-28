# Micro-PCB R03 binary class correction

Native README names contradict visible photos. Both reviewers inspected three different conditions for each code (39 images). Raspberry Pi identity is visible for G/H/M. Exact Raspberry Pi variants were not required or asserted for this single-class task. The override applies to this R03 workspace only.

| Code | Native README claim | Reviewed RPi? | Visible evidence |
|---|---|---|---|
| A | Raspberry Pi A+ | No | ELEGOO MEGA2560 text; blue Mega-shaped board; no Raspberry Pi identity |
| B | Arduino Mega 2560 (Blue) | No | MEGA text on black long microcontroller board |
| C | Arduino Mega 2560 (Black) | No | MEGA text; yellow headers on black long microcontroller board |
| D | Arduino Mega 2560 (Black and Yellow) | No | Arduino DUE text and logo on blue board |
| E | Arduino Due | No | BeagleBone text and dual long expansion headers |
| F | Beaglebone Black | No | ARDUINO UNO text and Arduino infinity logo |
| G | Arduino Uno (Green) | Yes | Raspberry Pi logo on green SBC with Ethernet and four USB ports |
| H | Raspberry Pi 3 B+ | Yes | Raspberry Pi logo on green SBC with Ethernet and four USB ports |
| I | Raspberry Pi 1 B+ | No | ESP8266 Wireless Module UNO text on Arduino-shaped board |
| J | Arduino Uno Camera Shield | No | ELEGOO UNO R3 text on black board |
| K | Arduino Uno (Black) | No | WiFi/ESP8266 module on blue UNO-shaped board; no Raspberry Pi identity |
| L | Arduino Uno WiFi Shield | No | ARDUINO LEONARDO text and Arduino infinity logo |
| M | Arduino Leonardo | Yes | Raspberry Pi logo on compact green SBC with single USB and no Ethernet |

3 different conditions for each of 13 source codes; not all images individually adjudicated

Earlier A/H/I binary ontology was visually invalid; no old split is considered a valid Raspberry Pi reference. New global source-family + SHA + pHash grouping uses seed42 70/15/15.

QUARANTINED_CLASS_MAPPING_CONFLICT; must not report historical RPi AP as a valid accuracy baseline

Review evidence: `contact_sheets/micro_ALL_CODES_1.png`, `_2.png`, `_3.png`; exact source coordinates/hashes in `micro_ALL_CODES_index.json`.

Prepared records: {'images': 2875, 'positive': 1875, 'negative': 1000, 'by_code': {'A': 100, 'B': 100, 'C': 100, 'D': 100, 'E': 100, 'F': 100, 'G': 625, 'H': 625, 'I': 100, 'J': 100, 'K': 100, 'L': 100, 'M': 625}}. Other-board native boxes remain provenance only and do not become RPi labels.
