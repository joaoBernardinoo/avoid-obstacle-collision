
::: mermaid
graph TD
    OD(ObstacleDetected) --> Dir(Direction)
    TV(TargetVisible) --> Dir
    Dir --> Act(Action)
    OD --> Act
    Act --> S(Success)
    TV --> S
:::