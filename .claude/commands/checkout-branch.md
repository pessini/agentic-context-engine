Switch to an existing branch by checking out its worktree, or creating one if needed.

**Arguments:** $ARGUMENTS should be the branch name (full or partial match)

**Examples:**
- `/checkout-branch feature/john/add-caching` → switch to branch (create worktree if needed)
- `/checkout-branch add-caching` → partial match, resolve to full branch name
- `/checkout-branch fix` → if multiple matches, list them and ask to be specific

**Steps:**
1. Parse branch name from arguments
2. Get all local branches: `git branch --format='%(refname:short)'`
3. Resolve branch name:
   - Exact match: use directly
   - Partial match: find branches containing the search term
   - No match: show error with similar branches (if any)
4. Get worktree list: `git worktree list --porcelain`
5. Check if resolved branch has an existing worktree
6. If worktree exists:
   - Show path and suggest `cd <path>`
7. If no worktree:
   - Construct worktree path: `../<sanitized-branch-name>` (replace all `/` with `-`)
   - Create worktree: `git worktree add <path> <branch>`
   - Show path and suggest `cd <path>`

**On success (worktree exists), output:**
```
✓ Branch already has worktree at: <worktree-path>

To switch to the worktree:
  cd <worktree-path>
```

**On success (worktree created), output:**
```
✓ Created worktree: <worktree-path>

To switch to the worktree:
  cd <worktree-path>
```

**Error handling:**
- Branch not found: "Branch not found: <name>. Did you mean one of these?" (list similar branches)
- Multiple partial matches: "Multiple branches match '<term>':" (list matches, ask to be more specific)
- No branches at all: "No branches found matching '<term>'"

**Branch resolution priority:**
1. Exact match on full branch name
2. Exact match on last segment (e.g., "add-caching" matches "feature/john/add-caching")
3. Partial substring match anywhere in branch name
