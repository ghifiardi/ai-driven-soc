# AI-SOC Agent Orchestration System

This folder contains the ritual workflow for implementing the AI-Driven SOC platform following the Research → Plan → Implement methodology.

## Folder Structure

```
.agents/
├── research.md          # Research phase template
├── plan.md             # Plan phase template  
├── progress.md         # Implementation progress tracking
├── decisions.md        # Architectural decisions log
├── risks.md           # Risk register
├── orchestrator_prompt.md  # System prompt for orchestrator
├── prompts/           # Child agent prompts
│   ├── research_prompt.md
│   ├── plan_prompt.md
│   └── implement_prompt.md
└── contracts/         # Agent contracts
    └── child_agent_schema.json
```

## Workflow

### 1. Research Phase
- Use `research.md` template
- Map files and symbols with roles
- Document execution flow
- Identify risks and evidence
- **Output**: Complete research document

### 2. Plan Phase  
- Use `plan.md` template
- Define goals and non-goals
- Specify file changes with line ranges
- Create execution sequence
- Plan rollback strategy
- **Output**: Detailed implementation plan

### 3. Implement Phase
- Follow the approved plan exactly
- Update `progress.md` after each step
- Run validation tests
- Handle issues and update risks
- **Output**: Implemented changes with progress tracking

## Key Principles

- **No code before approved plan**
- **Context hygiene** (≤40% model window)
- **Evidence pointers** (path + [Lx–Ly] + why_relevant)
- **Directed Restart** if plan drifts
- **Real-time progress tracking**

## Child Agents

- **Codebase Surveyor**: Research and analysis
- **Test Enumerator**: Testing and validation
- **Config Mapper**: Configuration and environment
- **Strategic Planner**: Planning and architecture
- **Implementation Executor**: Code implementation

## Usage

1. Copy the appropriate template for your phase
2. Fill in the required sections
3. Follow the exact format specified
4. Update progress.md during implementation
5. Use child agents for specialized tasks

## Quality Gates

- Research completeness before planning
- Plan approval before implementation
- Test validation after each step
- Progress documentation in real-time












