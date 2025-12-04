# GIT 

## GET BRANCH
+ Pull Existing Branch in Local
```
git fetch origin mytopic
git branch mytopic FETCH_HEAD
git checkout mytopic
```

## GIT PUSH
+ Push local branch to remote origin
  `git push -u origin branchname`

## GIT RESTORE
+ Git restore file to remote
`git restore --source origin/main filename`
+ Git restore file to commit
`git restore --source=commit_has -- filename`

## GIT PATCH
+ Create Patch for last commit
`git format-patch -1 commit_has --stdout > patch`
+ Create Patch for last 3 commits
`git format-patch HEAD~3 --stdout > patch`

# GIT DIFF
+ Diff staged and unstaged changes
  `git diff HEAD`

# GIT SEARCH 
+ Search in Git across branches, commits
`git rev-list --all | xargs git grep -F "KEY"`

+ Grep (also supports AND/OR)
  `git grep -E "regex_pattern" -n -l`
  `git grep "pattern" HEAD~1`
  
  

  
## Remove files
+ Remove untracked file
`git clean -f `
+ Remove tracked file change
`git restore filename`

# Untracked file
+ List untracked files
  `git ls-files --others --exclude-standard -z`
# GIT stash
```
git stash apply stash@{1}
git stash save  "STRING" 
git stash list
git stash show stash@{0} --patch 
git stash branch test_2 stash@{0}
```

# Git remote
+ Add remote
  `git remote add origin https://github.com/username/repository.git`

# GIT LOG
+ Git Long oneline color full
  `git log --pretty="%C(Yellow)%h  %C(reset)%ad (%C(Green)%cr%C(reset))%x09 %C(Cyan)%an: %C(reset)%s" --date=short `
+ Changes in Commit 
`git log -p`
+ Oneline per commit
`git log --oneline`

+ Range of dates
  `git log  --oneline --after "2020-04-12" --before "2020-04-014" --date=short  --pretty="%C(Yellow)%h  %C(reset)%ad (%C(Green)%cr%C(reset))%x09 %C(Cyan)%an: %C(reset)%s"`
+ Git graph
  ```
  git config --global alias.graph "log --oneline --graph --all --decorate"
  git graph
  ```
