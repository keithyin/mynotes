```{mermaid}
graph TB
  subgraph ps
    parameter_server
  end
  subgraph worker:
  gradients_queue
  worker0
  worker1
  worker2
  ...
  worker0 --> |gradient|gradients_queue
  worker1 --> |gradient|gradients_queue
  worker2 --> |gradient|gradients_queue
  ... --> |gradient|gradients_queue
  end

  parameter_server --> |variables|worker0
  parameter_server --> |variables|worker1
  parameter_server --> |variables|worker2
  parameter_server --> |variables|...

  gradients_queue --> |gradient|parameter_server
```
<center>Between-graph,同步</center>
```{mermaid}
graph LR

  subgraph worker:
    worker0
    worker1
    worker2
    ...
  end
  subgraph ps
    parameter_server
  end
  parameter_server --> |variables|worker0
  parameter_server --> |variables|worker1
  parameter_server --> |variables|worker2
  parameter_server --> |variables|...
  worker0 --> |gradient|parameter_server
  worker1 --> |gradient|parameter_server
  worker2 --> |gradient|parameter_server
  ... --> |gradient|parameter_server
```
<center>Between-graph,异步</center>

```{mermaid}
graph LR
  subgraph worker:
    worker0
    worker1
    worker2
    ...
  end
  subgraph supervisor
    Supervisor
  end
  Supervisor --> |monitor|worker0
  Supervisor --> |monitor|worker1
  Supervisor --> |monitor|worker2
  Supervisor --> |monitor|...
```
```{mermaid}
graph LR
  subgraph worker:
    worker0
    worker1
    worker2
    ...
  end
  subgraph supervisor
    Supervisor
  end
  Supervisor --> |monitor|worker0
  Supervisor --> |monitor|worker1
  Supervisor --> |monitor|worker2
  Supervisor --> |monitor|...
```
