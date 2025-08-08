# LAiSER Contributor's Guide: Conventional Commits

This guide provides a simple overview of the Conventional Commits specification. Following these guidelines for your commit messages when contributing to any of the LAiSER projects. This is to improve readability, automate changelog generation, and help streamline the release process.

## The Conventional Commit Message Format

A conventional commit message follows a simple, structured format:

```shell
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

#### 1. Type

The type is a mandatory part of the commit message and it defines the category of the change. Here are the most common types:

- feat: A new feature for the user.
- fix: A bug fix for the user.
- chore: Routine tasks, maintenance, or dependency updates. No production code changes.
- docs: Documentation changes only.
- style: Code style changes (e.g., formatting, white-space). No functional code changes.
- refactor: A code change that neither fixes a bug nor adds a feature.
- test: Adding missing tests or correcting existing tests.

#### 2. Scope (Optional)

The scope provides additional context for the commit. It's an optional noun that specifies the section of the codebase affected by the change.

- Example: `feat(api): add new user endpoint`
- Example: `fix(ui): correct button alignment on login page`

#### 3. Description

The description is a short, imperative summary of the code changes.

- Do: `add new feature`
- Don't: `added new feature or adds new feature`

#### 4. Body (Optional)

The body is used to provide more detailed information about the changes. It should explain the "what" and "why" of the commit, not the "how."

#### 5. Footer (Optional)

The footer is used for two main purposes:

- Breaking Changes: Any commit that introduces a breaking API change must start the footer with `BREAKING CHANGE:`, followed by a description of the change.
- Referencing Issues: To reference issues, use keywords like Closes #123 or Fixes #456.

### Examples
#### Commit with a new feature

```shell
feat: allow users to upload profile pictures

Users can now upload a JPG or PNG file to be used as their profile picture.
This change includes new UI elements and backend logic to handle the upload.
```


#### Commit with a bug fix and closing an issue
```shell
fix: prevent crash when user clicks the save button twice

The application was crashing due to a race condition when the save button
was clicked multiple times in quick succession. This has been resolved by
disabling the button after the first click.

Closes #78
```

#### Commit with a breaking change

```shell
refactor: switch to new authentication API

BREAKING CHANGE: The old authentication endpoint `/auth/login` is now
deprecated. All clients must be updated to use the new `/api/v2/auth/token`
endpoint. The request and response formats have also changed.
```

## Releases and Semantic Versioning

Semantic Versioning is a versioning system that uses a three-part number to communicate the kinds of changes in a release: MAJOR.MINOR.PATCH. By using Conventional Commits, we can automate how we determine the next version number.
How It Works

Here’s how commit types relate to versioning:

- PATCH version (e.g., 1.0.0 -> 1.0.1)

    - Incremented for backward-compatible bug fixes.

    - Corresponds to fix commits.

- MINOR version (e.g., 1.0.1 -> 1.1.0)

    - Incremented for new, backward-compatible functionality.

    - Corresponds to feat commits.

- MAJOR version (e.g., 1.1.0 -> 2.0.0)

    - Incremented for any incompatible API changes (breaking changes).

    - Corresponds to any commit that includes a BREAKING CHANGE: in the footer.

### The Connection

When we create a release, we can look at the commit history since the last release:

- If there are any BREAKING CHANGE: commits, we make a MAJOR release.

- If there are feat commits but no breaking changes, we make a MINOR release.

- If there are only fix commits (and other types like chore, docs, etc.), we make a PATCH release.

This direct link between our commit messages and versioning makes our release process predictable and clear for everyone using our software.

### Examples

#### Commit with a new feature (leads to a MINOR release)

```shell
feat: allow users to upload profile pictures

Users can now upload a JPG or PNG file to be used as their profile picture.
This change includes new UI elements and backend logic to handle the upload.
```

#### Commit with a bug fix (leads to a PATCH release)

```shell
fix: prevent crash when user clicks the save button twice

The application was crashing due to a race condition when the save button
was clicked multiple times in quick succession. This has been resolved by
disabling the button after the first click.

Closes #78
```

#### Commit with a breaking change (leads to a MAJOR release)

```shell
refactor: switch to new authentication API

BREAKING CHANGE: The old authentication endpoint `/auth/login` is now
deprecated. All clients must be updated to use the new `/api/v2/auth/token`
endpoint. The request and response formats have also changed.
```

By adhering to these conventions, our project's commit history will be more organized, and we can automate our release and changelog processes effectively.