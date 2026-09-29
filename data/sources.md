# Data Sources & Real-World Context Documentation

## 1. Primary Real-World Context: Jinnah University for Women (JUW)

* **Institution**: Jinnah University for Women (JUW), North Nazimabad, Karachi, Pakistan.
* **Official Portals**:
  * Main University Portal: [https://www.juw.edu.pk/](https://www.juw.edu.pk/)
  * Department of Computer Science & Software Engineering: [https://cs.juw.edu.pk/](https://cs.juw.edu.pk/)

### Documented Computer Laboratory & Workstation Facilities
According to official JUW documentation:
1. **Departmental CS & SE Labs**: Multiple dedicated computer laboratories equipped for Computer Science, Software Engineering, Data Science, and Artificial Intelligence coursework.
2. **Library Internet & Digital Research Lab**: Workstation units provided for digital library access (JLMS and HEC Digital Library) and student research.
3. **JUBIC Incubation Workstations**: Fully-equipped computer workstations situated in the Jinnah University Business Incubator (JUBIC).

---

## 2. University-Specific Data Strategy (JUW Real Images)

* **Folder Reserved**: `dataset/raw/juw_real/`
  * `dataset/raw/juw_real/empty/` (For empty seat / workstation crops)
  * `dataset/raw/juw_real/occupied/` (For occupied seat / workstation crops)
* **Real-World Integration Plan**:
  * Real campus lab photographs collected on-site at JUW are deposited directly into `dataset/raw/juw_real/`.
  * The automated preprocessing pipeline (`src/prepare_dataset.py`) scans this directory, crops/normalizes workstation regions, and incorporates them into validation, testing, and fine-tuning datasets seamlessly.

---

## 3. Initial Training & Validation Dataset Strategy

To construct an immediate, fully functional, and verifiable CNN pipeline prior to on-site photo collection:
* **Dataset Structure**: 
  * Class 0: `EMPTY` (Unoccupied seat / workstation with empty chair, desk surface, monitors turned off or idle).
  * Class 1: `OCCUPIED` (Occupied seat / workstation with student seated, active laptop/PC usage, or workspace presence).
* **Sample Count**: 600 total curated image patches (300 EMPTY, 300 OCCUPIED).
* **Data Splits**:
  * **Train Set**: 70% (420 samples: 210 Empty, 210 Occupied)
  * **Validation Set**: 15% (90 samples: 45 Empty, 45 Occupied)
  * **Test Set**: 15% (90 samples: 45 Empty, 45 Occupied)
* **Data Leakage Mitigation**: Independent spatial sampling ensures zero overlap between crop sources across training, validation, and testing sets.
