# Product Guidelines

## Interaction Principles (ChatMLOps)
- **Tone:** Helpful and Conversational. The assistant should act as a knowledgeable guide, simplifying complex ML concepts while maintaining technical accuracy. It should feel like a pair programmer helping the user navigate the lifecycle.
- **Responsiveness:** Prioritize clear, immediate feedback. If a process takes time (like training), the system should explain what is happening rather than staying silent.

## Visual & Presentation Standards
- **Terminal Output:** Maximize visual engagement using rich terminal features. Utilize colors (e.g., green for success, red for errors, yellow for warnings) and progress bars (`tqdm`) for long-running operations.
- **Reporting:** When displaying tables (e.g., model comparison results), ensure they are well-formatted and easy to read at a glance during a live demo.

## Documentation & Planning
- **Detail Level:** Balanced Approach.
    - **High-Level:** All plans and documentation should start with a clear, executive-summary style overview of the goal and expected outcome.
    - **Technical Depth:** Sub-tasks and implementation steps must provide specific technical details, including relevant file paths, class names, and function signatures, catering to the Data Scientist audience.
