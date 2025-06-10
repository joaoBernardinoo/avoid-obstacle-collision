
::: mermaid
graph TD
    D[DistanciaDiscretizada] --> OD(ObstacleDetected)
    A[AngToObjectDiscretizado] --> OD
    D --> TV(TargetVisible)
    A --> TV
    OD --> Dir(Direction)
    TV --> Dir
    Dir --> Act(Action)
    OD --> Act
    Act --> S(Success)
    TV --> S
:::