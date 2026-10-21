# Data Checksums — Baseline Fingerprints

These SHA-256 checksums are the **baseline fingerprints** of the six original project files. They are used to verify that the raw datasets, rubric and course template remain unchanged throughout the project. Any future check should recompute the SHA-256 of each file and compare it with the value below: identical = file unchanged; different = file has been modified or corrupted.

- Baseline recorded: 20 October 2026 (Phase 1 audit)
- Re-verified: 20 October 2026 (Phase 2, Step 2.1) — all six **MATCH**
- Folder: `C:\Users\likes\OneDrive\Documents\Production_Pulse_Anamoly_Detection`

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `TEP_FaultFree_Training.RData` | 24,678,017 | `12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940` |
| `TEP_FaultFree_Testing.RData` | 47,327,663 | `4f45afafa469793eeb7203fb9ed10ed0b1724c73c9c95537f15a0889ade0ebd4` |
| `TEP_Faulty_Training.RData` | 494,063,194 | `058df8bd90c5f575f44d2f6085bdd3bc3662e3e9cc25a50e34af42be3748952b` |
| `TEP_Faulty_Testing.RData` | 836,882,037 | `c2f1aabc77498efda4d5f9ad81ecc829e2fcd98752b84cf5b2a06558cabacc36` |
| `Mark Split 21CSE732P rubrics .pdf` | 96,282 | `88ee341479b235f13849299f479b44eb684072bbf6547825fc4106a2d1571e0b` |
| `Applied AI Industry Worklets Student Template.xlsx` | 47,472 | `bd58307d70415086a3b4c0f274938e6e85ceb4db5a35f98b1a1afbf4dd453013` |

## How to re-check on Windows (PowerShell, from the project folder)
```powershell
Get-FileHash -Algorithm SHA256 *.RData, "Mark Split 21CSE732P rubrics .pdf", "Applied AI Industry Worklets Student Template.xlsx" | Format-Table Hash, Path -AutoSize
```
Windows prints hashes in UPPERCASE; compare them ignoring case.
