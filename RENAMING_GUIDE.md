# Project Renaming Guide: `benchmark_tool` → `dense_slam_benchmark_tool`

## Overview
Renaming involves multiple layers: project metadata, Python package names, imports, directory structure, and documentation.

---

## 1. Renaming Strategy Options

### Option A: Conservative (Recommended for your case)
**Best for:** Minimal disruption, existing development workflows

```
Project name (pyproject.toml):  benchmark_tool → dense_slam_benchmark_tool
Package name (import):          benchmark → benchmark (KEEP AS IS)
Root directory:                 benchmark_tool → dense_slam_benchmark_tool (optional)
Description:                    Update to reference SLAM/depth estimation
```

**Pros:**
- Only change pyproject.toml and README
- All Python imports remain unchanged
- Scripts, notebooks, dependencies work without modification
- Easy to maintain existing development workflow
- Low risk of breaking things

**Cons:**
- "dense_slam_benchmark_tool" project name vs "benchmark" package is slightly asymmetric

### Option B: Full Refactor (Comprehensive)
**Best for:** Clean slate, fresh repository structure

```
Project name (pyproject.toml):  benchmark_tool → dense_slam_benchmark_tool
Package name (import):          benchmark → dense_slam_benchmark
Root directory:                 benchmark_tool → dense_slam_benchmark_tool
All imports:                    from benchmark... → from dense_slam_benchmark...
```

**Pros:**
- Fully aligned naming
- Clear SLAM focus in package name
- Professional appearance

**Cons:**
- ~20-30 import updates across codebase
- Need to update all files in `benchmark/` → `dense_slam_benchmark/`
- Risk of missed imports causing subtle bugs
- Higher refactoring effort

---

## 2. Recommendation

### **Go with Option A (Conservative)** for these reasons:

1. **Current Codebase Status:** You're actively developing, have multiple working scripts
2. **Import Pattern:** 20+ import statements across code would need updates with Option B
3. **External Dependencies:** `external/` submodules reference internal imports
4. **Submodules:** `thirdparty/` projects import from `benchmark` package
5. **Minimal Value:** Package name change doesn't affect functionality

### **What to Change (Option A):**

#### `pyproject.toml`
```toml
# Change this:
name = "benchmark_tool"
description = "A reconstruction benchmark tool"

# To this:
name = "dense_slam_benchmark_tool"
description = "Dense SLAM & Depth Estimation Benchmark Tool - Unified evaluation framework for dense 3D reconstruction and depth estimation methods"
```

#### `README.md`
- Use new project name in title
- Update installation instructions if needed
- Update create environment examples

#### Directory (Optional but Recommended)
```
# Before:
/home/spiderman/vscode_projects/benchmark_tool/

# After:
/home/spiderman/vscode_projects/dense_slam_benchmark_tool/
```

#### Internal Package Name
**Keep `benchmark/` folder as-is:**
```python
from benchmark.dataloader import CameraDataset  # ← No changes needed
from benchmark.metrics import evaluate          # ← No changes needed
```

**Why:** The Python package name can differ from the project name. Example:
- Project: `django-rest-framework` (PyPI name)
- Package: `rest_framework` (import name)

---

## 3. If You Later Want Option B (Full Refactor)

You can always do this refactoring in the future in a single batch:

**Bulk Rename Steps:**
1. Rename folder: `benchmark/` → `dense_slam_benchmark/`
2. Update all imports: `sed -i 's/from benchmark/from dense_slam_benchmark/g' **/*.py`
3. Update pyproject.toml: `packages = ["dense_slam_benchmark"]`
4. Run tests to verify

**Best Time:** When you stabilize the project (after next benchmark cycle)

---

## 4. My Recommendation: Hybrid Approach

**Phase 1 (Today):**
1. Update `pyproject.toml`: `name` and `description`
2. Optional: Rename root directory
3. Update `README.md` with new name and SLAM focus
4. Keep `benchmark/` package name unchanged

**Phase 2 (After current work stabilizes):**
- Consider full refactor to `dense_slam_benchmark` package if it bothers you
- Can be done in one afternoon with find-replace

---

## 5. Specific Changes Needed for Phase 1

### `pyproject.toml`
```toml
[project]
name = "dense_slam_benchmark_tool"
version = "0.0.1"
authors = [
  { name="PengZai", email="649365461@qq.com" }
]
description = "Dense SLAM & Depth Estimation Benchmark Tool - Unified framework for evaluating depth estimation and 3D reconstruction methods on standardized datasets"
readme = "README.md"
requires-python = ">=3.10.0"

# ... rest unchanged
```

### `README.md`
- Title: `# Dense SLAM Benchmark Tool`
- Update installation section
- Mention the three core functions explicitly

### `.gitignore` (if it references old name)
- Check and update paths if any

### No other changes required!

---

## 6. Why This Approach is Smart

| Aspect | Option A | Option B |
|--------|----------|---------|
| **Effort** | 10 minutes | 2-3 hours |
| **Risk** | Minimal | Medium |
| **Breakage Potential** | None | Moderate |
| **Time to Resume Development** | Immediate | 30 min+ debugging |
| **Future Flexibility** | Can upgrade anytime | N/A |
| **Value Added Now** | High (clear naming, updated docs) | Medium (alignment) |

---

## 7. Implementation Checklist (Option A)

- [ ] Update `pyproject.toml` `name` field
- [ ] Update `pyproject.toml` `description` field
- [ ] Rename root directory if desired (git mv)
- [ ] Update `README.md` title and description
- [ ] Update installation instructions in README
- [ ] Test: `pip install -e .` still works
- [ ] Verify: `python -c "from benchmark import metrics"` still works
- [ ] Update `.gitmodules` paths if needed
- [ ] Commit with message: "Rename project to dense_slam_benchmark_tool"

---

## Questions to Consider

1. **Do you want to rename the root directory?** 
   - Recommended: Yes (git mv benchmark_tool dense_slam_benchmark_tool)
   - Allows clean fresh start

2. **Will this be released on PyPI?**
   - If yes: Keep clean `pyproject.toml` name
   - If no: Less critical (internal project)

3. **Do others clone this repo?**
   - If yes: Consider updating clone path in docs
   - If no: Only you need to update local clone

---

## Conclusion

**Recommended Action:** 
Implement **Option A** today (5 min update to metadata + README).  
Plan **Option B** refactoring for future when code stabilizes.

This gives you:
- ✅ Clear new project name externally
- ✅ SLAM focus signaled in metadata  
- ✅ Zero disruption to current work
- ✅ Flexibility to refactor imports later if desired
