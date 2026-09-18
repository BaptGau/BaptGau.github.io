---
layout: default
title: AI Engineering
---
---
layout: default
title: "AI Engineering"
description: "Thoughts on how to design AI architectures"
author: "Baptiste Gautier"
---

*By Baptiste Gautier*

# TL;DR

Production AI engineering is an architecture-design problem. This article presents a framework for decomposing a product feature ($F$) into operational tasks:

$$
F \longrightarrow T={t_1,\ldots,t_n},
$$

and assigning those tasks to estimators such as language models, classifiers, retrieval systems, heuristics, or deterministic code.

Estimators are composed into an architecture:

$$
A=(P,E,G,V,R,\Phi),
$$

which defines task grouping, estimator assignment, execution dependencies, validation, retries, and fallbacks.

Architectures should be compared using total expected cost:

$$
\mathbb E  
\left[  
\mathcal C_{\mathrm{direct}}(A)+\Lambda(A)  
\right],
$$

where $\mathcal C_{\mathrm{direct}}$ includes execution and recovery costs, while $\Lambda$ captures the consequences of invalid, delayed, incomplete, or harmful results.

Grouping tasks can reduce context duplication and request overhead but may couple failures and increase recovery costs. Modular execution improves specialization and failure isolation but may duplicate computation. Neither is universally superior.

The objective is therefore to find a **golden architecture**: the composition of tasks, estimators, validators, execution paths, retries, and fallbacks that satisfies the required service constraints at the lowest sustainable total cost. This article explores how each of these architectural components can be optimized toward that objective.

# Introduction

Building a production feature with Large Language Models requires more than selecting a capable model and writing an effective prompt. The model call is only one component of a larger system that must transform uncertain outputs into a reliable product experience.

Like classical machine-learning systems, LLM applications face distribution shift, imperfect offline evaluation, and differences between experimental and production behavior. LLMs add further architectural challenges: they can perform several heterogeneous tasks in one invocation, generate outputs over a broad and weakly constrained space, and exhibit failures that are difficult to detect through ordinary type checking alone.

These properties create a deceptively simple design choice. An engineer can ask one general-purpose model to classify, extract, translate, explain, and format a response in a single call. This approach shares context and may minimize direct token or compute cost. Alternatively, the engineer can separate those responsibilities across specialized models, retrieval systems, validators, and deterministic functions. This modular approach may cost more on the successful path but can improve observability, failure isolation, and recovery.

# Table of contents
{:.no_toc}

* TOC
{:toc}

# 1. From Features to Operational Tasks

The AI engineering process begins with product-level decomposition. A broad product feature $F$ must be translated into a finite set of explicitly defined operational tasks:

$$
F \longrightarrow T=\{t_1,\ldots,t_n\}.
$$

Each task $t_i$ represents a bounded operational objective with a defined interface. Tasks may depend on one another and may later be executed sequentially, concurrently, or conditionally as part of a larger architecture.

Many tasks are AI-driven, such as detecting sentiment, classifying text, performing semantic retrieval, or extracting verbatim quotations that support an assertion. Others are deterministic software operations. A task might parse generated text into a strict data structure, verify that an extracted quotation occurs in a source document, or map a predicted boolean value to a user-interface label.

For each task $t_i$, the system architect should define:

- its input space $\mathcal{X_i}$ and output space $\mathcal Y_i$;
- its semantic and formatting requirements;
- its latency, reliability, and cost targets;
- the consequences of failure;
- whether partial output, abstention, or escalation is acceptable.

The conditions governing output validity can be represented as a constraint set:

$$
C_i=\{c_{i,1},\ldots,c_{i,m_i}\}.
$$

Examples include:

- **Schema compliance:** the output follows a specified JSON schema or Markdown structure;
- **Type validity:** the output has the expected data type;
- **Length boundaries:** the output remains within defined token or character limits;
- **Grounding:** an extracted quotation occurs in the supplied source;
- **Semantic correctness:** the output expresses the required meaning or classification.

For an input $x\in\mathcal X_i$ and output $y\in\mathcal Y_i$, the $k$-th constraint is represented as a binary predicate:

$$
c_{i,k}(x,y)\in\{0,1\}.
$$

The task-level validity function is then:

$$
S_i(x,y) = \prod_{k=1}^{m_i}c_{i,k}(x,y).
$$

Consequently,

$$
S_i(x,y)=1 \iff \forall k,\;c_{i,k}(x,y)=1.
$$

Some constraints, such as schema or type validity, can be evaluated deterministically. Others, such as factual correctness or semantic relevance, may require human judgment or an imperfect model-based evaluator. The formal predicate represents the desired ground truth; its production implementation may only approximate that truth.

Decomposing a feature into operational tasks is therefore both a product decision and a systems-design decision. Misspecification at this stage propagates through the architecture: estimators, evaluation datasets, validators, and recovery policies may all be optimized against the wrong objective. Explicit task definitions establish the contracts against which alternative estimators and architectures can later be compared.

# 2. Estimators as Execution Components

Once operational tasks and their success conditions have been defined, the next question is *how those tasks should be executed*. We call the mechanism responsible for attempting one or more tasks an **estimator**.

An estimator $e$ receives an input $x\in\mathcal X$ and induces a conditional probability distribution over an output space $\mathcal Y$:

$$
Y\sim P_e(\cdot\mid x;\theta_e),
$$

where $Y$ is the generated output and $\theta_e$ represents the estimator’s complete configuration.

This probabilistic formulation accommodates both stochastic and deterministic systems. For a generative model, repeated executions on the same input may produce different outputs following a specific latent distribution. For a deterministic function $f_e$, the distribution collapses to a Dirac measure concentrated on a single output:

$$
P_e(\cdot\mid x;\theta_e) = \delta_{f_e(x)}.
$$

## 2.1. Estimator Configuration

The configuration $\theta_e$ depends on the type of estimator. For a text model-based estimator, a common parameterization is:

$$
\theta_e=(\pi,\rho,s),
$$

where:

- **The underlying model $\pi$:** the model (often LLM or encoder based) responsible for processing the task, including its specific version or checkpoint;
- **The prompt template $\rho$:** the instructions, context construction, and examples supplied to the model;
- **The inference settings $s$:** parameters such as temperature, top-p, maximum output length, decoding strategy, or reasoning budget.

For systems that do not use prompting or stochastic decoding, $\rho$ or $s$ may be set to $\varnothing$.

The triplet $(\pi,\rho,s)$ is particularly useful for describing language-model estimators, but it is not exhaustive for every estimator type. A retrieval estimator may additionally depend on an embedding model, an index, a similarity function, filtering rules, … A deterministic estimator may instead be defined by a software implementation and its configuration.

The more general notation $\theta_e$ allows these estimator-specific parameters to be represented without forcing every execution mechanism into an LLM-specific abstraction.

## 2.2. Estimator Types

As discussed in §2.1, an estimator is not necessarily a generative LLM. Depending on the task, an estimator may be:

- a Large Language Model;
- a fine-tuned Small Language Model;
- an encoder based classifier;
- a semantic retrieval pipeline;
- a rule-based heuristic;
- a deterministic application function;
- a human decision process.

For example, sentiment classification might initially be performed by a general-purpose language model and later migrated to a specialized classifier. A grounding task might use lexical matching rather than generation. JSON serialization may be performed by deterministic application code rather than delegated to a model.

These mechanisms differ substantially in cost, latency, and behavior, but they share the same architectural role: each receives an input and attempts to produce an output that satisfies the task’s success conditions.

## 2.3. Task-Level Estimator Success

For a task $t_i$, let $\mathcal D_i$ denote the distribution of real-world inputs encountered by that task. Let $S_i(x,Y)$ be the task-validity function defined in Section 1:

$$
S_i(x,Y)=1
$$

if and only if the output satisfies every required constraint.

The success probability of estimator $e$ on task $t_i$ is:

$$
p_i(e) = \mathbb E_{x\sim\mathcal D_i} \left[ \Pr_{Y\sim P_e(\cdot\mid x;\theta_e)} \left( S_i(x,Y)=1 \right) \right].
$$

Equivalently, this can be written as a joint probability over production inputs and estimator outputs:

$$
p_i(e) = \Pr_{\substack{x\sim\mathcal D_i\\ Y\sim P_e(\cdot\mid x;\theta_e)}} \left( S_i(x,Y)=1 \right).
$$

This quantity depends on four elements:

1.  the production input distribution $\mathcal D_i$;
2.  the estimator family and implementation;
3.  the estimator configuration $\theta_e$;
4.  the task-validity function $S_i$.

Note that an estimator should not be described as universally “good” or “bad.” It can only be evaluated relative to a particular task, constraint set, configuration, and input distribution. This accounts for the performance gap between synthetic evaluation and production: when the input distribution shifts, performance shifts with it.

## 2.4. Estimators Executing Multiple Tasks

An estimator may execute one task or a group of related tasks. Let $B_g\subseteq T$ be a group of tasks assigned to estimator $e_g$. The estimator then produces a structured output:

$$
Y_g=\{Y_{g,i}\}_{t_i\in B_g},
$$

where $Y_{g,i}$ is the portion of the output corresponding to task $t_i$.

The estimator succeeds on the complete task group only when every required task succeeds:

$$
S_{B_g}(x,Y_g) = \prod_{t_i\in B_g} S_i(x_i,Y_{g,i}).
$$

Its joint success probability is therefore:

$$
p_{B_g}(e_g) = \mathbb E_{x\sim\mathcal D_{B_g}} \left[ \Pr\left( S_{B_g}(x,Y_g)=1 \mid x,e_g \right) \right].
$$

The tasks may share the same input, reasoning process, or generated structure, making their success events dependent.

Grouping tasks can create efficiencies by sharing context and computation. It can also introduce interference, coupled failures, and more complex validation requirements. Determining which tasks should share an estimator is therefore not merely an estimator-selection problem; it is an architectural decision. Section 4 provides tools to help diagnose those effects.

The next section introduces the cornerstone of this work: The notion of *architecture* and its components.

# 3. Architectures as Compositions of Estimators

Estimators perform the execution of individual tasks or task groups, but they do not operate in isolation. In production, they are embedded within a broader architecture that determines how tasks are grouped, which estimators execute them, how information moves between components, how outputs are validated, and how the system responds to failure.

We represent an architecture as:

$$
A=(P,E,G,V,R,\Phi),
$$

where:

- $P$ partitions tasks into groups for the primary execution path;
- $E$ assigns an estimator to each task group;
- $G$ defines execution order and dependencies;
- $V$ contains output validators;
- $R$ defines retry and repair policies;
- $\Phi$ defines fallback and escalation policies.

To illustrate these components, consider a support-ticket feature decomposed into four operational tasks:

$$
T=\{t_1,t_2,t_3,t_4\},
$$

where:

- $t_1$: detect the language;
- $t_2$: classify sentiment;
- $t_3$: classify urgency;
- $t_4$: extract a supporting quotation.

## 3.1. Partitioning Tasks into Estimator Groups

The partition $P$ determines which tasks are executed together and which are isolated.

Formally:

$$
P=\{B_1,\ldots,B_q\},
$$

where each $B_g$ is a nonempty subset of $T$ satisfying:

$$
B_g\cap B_h=\varnothing \qquad \text{for }g\neq h,
$$

and:

$$
\bigcup_{g=1}^{q}B_g=T.
$$

Every task therefore belongs to exactly one group in the architecture’s primary execution path.

For the support-ticket feature, one possible partition is:

$$
P= \left\{ \{t_1\}, \{t_2,t_3\}, \{t_4\} \right\}.
$$

This partition isolates language detection and quotation extraction while grouping sentiment and urgency classification.

The grouping reflects an architectural hypothesis. Sentiment and urgency are both about performing classification, so executing them together may allow the estimator to share context and reasoning. Quotation extraction has a different purpose: its output must be grounded in the source text. Isolating it allows the system to apply a specialized prompt, a deterministic grounding validator, and a targeted retry policy.

Two limiting cases are possible.

In a fully modular architecture, every task receives its own group:

$$
P_{\mathrm{modular}} = \left\{ \{t_1\}, \{t_2\}, \{t_3\}, \{t_4\} \right\}.
$$

In a fully grouped architecture, every task is assigned to one group:

$$
P_{\mathrm{grouped}} = \left\{ \{t_1,t_2,t_3,t_4\} \right\}.
$$

Most production architectures fall between these extremes.

The partition influences:

- context duplication and token consumption;
- opportunities for model specialization;
- possible task interference;
- failure isolation;
- retry granularity;
- opportunities for concurrent execution;
- the scope of recomputation after failure.

The partition describes the primary task assignment. An architecture may still execute a task through additional estimators during validation, ensemble voting, speculative execution, retry, or fallback.

## 3.2. Assigning Estimators to Task Groups

Once the partition has been defined, the architecture assigns an estimator to each task group:

$$
E=\{e_g\}_{g=1}^{q},
$$

where:

$$
e_g\in\mathcal E(B_g).
$$

Here,

$$
\mathcal E(B_g)
$$

denotes the set of estimators capable of executing the tasks contained in group $B_g$.

For the support-ticket partition:

$$
E(\{t_1\})=e_{\mathrm{language}}, \quad E(\{t_2,t_3\})=e_{\mathrm{classification}}, \quad E(\{t_4\})=e_{\mathrm{quote}}.
$$

Each estimator has its own configuration:

$$
e_g=e(\cdot;\theta_g).
$$

For a text model-based estimator, this configuration may be expressed as:

$$
\theta_g=(\pi_g,\rho_g,s_g),
$$

where $\pi_g$ is the underlying model, $\rho_g$ is the prompt, and $s_g$ contains the inference settings.

As discussed in §2.2, estimators assigned to groups can be of different type. In this example:

- $e_{\mathrm{language}}$ might be a lightweight language classifier;
- $e_{\mathrm{classification}}$ might be a small instruction-tuned language model;
- $e_{\mathrm{quote}}$ might be an extractive model or a constrained LLM;
- final output serialization might be performed by deterministic code.

Note that different estimators may use the same underlying foundation model while differing in their prompts or inference settings. This is quite a common practice with LLMs:

$$
\pi_g=\pi_h,
$$

Modularity therefore refers to the separation of task execution, not necessarily to the use of different models.

The estimator assignment determines which mechanism initially attempts each task group. It does not by itself determine execution order, validation, retry behavior, or fallback behavior.

## 3.3. Execution Order and Dependencies

The execution graph $G$ describes how information and control move through the architecture.

We represent the primary execution path as a directed graph:

$$
G=(N,D),
$$

where:

- $N$ is the set of execution nodes;
- $D\subseteq N\times N$ is the set of directed dependencies.

Nodes may represent:

- estimator invocations;
- deterministic transformations;
- validation operations;
- routing decisions;
- aggregation or join operations;
- output formatting.

An edge:

$$
n_a\rightarrow n_b
$$

means that execution of $n_b$ depends on the completion, output, or status of $n_a$.

The primary execution graph is normally acyclic. Retries and fallbacks may cause the runtime execution trace to revisit, replace, or bypass nodes, but those behaviors are defined separately through $R$ and $\Phi$.

For example, the support-ticket architecture might use the following graph:

<figure id="support-ticket-architecture" style="margin: 2rem auto; max-width: 760px;">
  <svg viewBox="0 0 760 660" role="img" aria-labelledby="flowchart-title flowchart-desc" xmlns="http://www.w3.org/2000/svg" style="display:block; width:100%; height:auto;">
    <title id="flowchart-title">Support-ticket architecture execution graph</title>
    <desc id="flowchart-desc">A support ticket branches to a language estimator and a quotation estimator. The language estimator feeds sentiment and urgency classification. Both branches join before schema validation and the final response.</desc>
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#2356a8" />
      </marker>
      <style>
        .node { fill: #f3f6fb; stroke: #2356a8; stroke-width: 2; }
        .edge { fill: none; stroke: #2356a8; stroke-width: 2.5; marker-end: url(#arrowhead); }
        .label { fill: #172033; font: 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-anchor: middle; }
      </style>
    </defs>

    <rect class="node" x="270" y="20" width="220" height="56" rx="8" />
    <text class="label" x="380" y="54">Support ticket</text>

    <rect class="node" x="60" y="130" width="250" height="56" rx="8" />
    <text class="label" x="185" y="164">Language estimator</text>

    <rect class="node" x="450" y="130" width="250" height="56" rx="8" />
    <text class="label" x="575" y="164">Quotation estimator</text>

    <rect class="node" x="60" y="250" width="250" height="72" rx="8" />
    <text class="label" x="185" y="280">
      <tspan x="185" dy="0">Sentiment and urgency</tspan>
      <tspan x="185" dy="22">estimator</tspan>
    </text>

    <rect class="node" x="270" y="380" width="220" height="56" rx="8" />
    <text class="label" x="380" y="414">Join results</text>

    <rect class="node" x="270" y="490" width="220" height="56" rx="8" />
    <text class="label" x="380" y="524">Schema validation</text>

    <rect class="node" x="270" y="600" width="220" height="56" rx="8" />
    <text class="label" x="380" y="634">Final response</text>

    <path class="edge" d="M330 76 L225 130" />
    <path class="edge" d="M430 76 L535 130" />
    <path class="edge" d="M185 186 L185 250" />
    <path class="edge" d="M225 322 L330 380" />
    <path class="edge" d="M535 186 L430 380" />
    <path class="edge" d="M380 436 L380 490" />
    <path class="edge" d="M380 546 L380 600" />
  </svg>
  <figcaption style="text-align:center; color:#586069; font-size:0.95rem; margin-top:0.6rem;">
    Example execution graph for the support-ticket architecture.
  </figcaption>
</figure>

In this architecture:

- language detection executes first;
- its result may determine the prompt, model, or parameters used for sentiment and urgency classification;
- quotation extraction executes concurrently because it only requires the original ticket;
- the join operation waits for the classification and quotation branches;
- schema validation occurs after aggregation;
- the final response is returned only after successful validation.

If all estimators depend only on the original input, they can execute concurrently:

$$
e_{\mathrm{language}}(x) \parallel e_{\mathrm{classification}}(x) \parallel e_{\mathrm{quote}}(x).
$$

Consequently, a modular partition does not necessarily imply high end-to-end latency. Independent estimators may execute in parallel, while a single grouped estimator may become a latency bottleneck if it produces a long or computationally demanding response.

The execution graph affects:

- critical-path latency;
- opportunities for concurrency;
- the amount and type of context exposed to downstream estimators;
- propagation of upstream errors;
- availability of intermediate representations;
- whether partial results can be returned;
- which operations must be repeated after failure.

The graph therefore captures both data flow and operational coupling. Two architectures may use the same task partition and estimators but exhibit different reliability and latency because their execution graphs differ.

## 3.4. Validators

Validators determine whether an intermediate or final output is acceptable.

Section 1 introduced $c_{i,k}(x,y)$ as the $k$-th validity condition associated with a task $i$. A validator is the production mechanism used to evaluate or approximate that condition.

A simple validator returns a binary decision:

$$
v_g(x,y_g)\in\{0,1\}.
$$

A more informative validator may return structured information:

$$
v_g(x,y_g) = (\widehat S_g,z_g,f_g),
$$

where:

- $\widehat S_g\in\{0,1\}$ is the validator’s observed pass-or-fail decision;
- $z_g$ identifies the detected error type;
- $f_g$ contains feedback that may support repair or escalation.

The distinction between the true constraint and its validator is important:

$$
v_{i,k}(x,y)\approx c_{i,k}(x,y).
$$

For deterministic conditions, the validator may implement the constraint exactly:

$$
v_{i,k}(x,y)=c_{i,k}(x,y).
$$

For semantic conditions, the validator may only provide an imperfect approximation and can produce false acceptances or false rejections.

For example, a structural language validator may verify that the output is an allowed language code:

$$
v_{\mathrm{language}}^{\mathrm{struct}}(x,y) = \mathbf 1 \left[ y\in \{\text{en},\text{fr},\text{de},\ldots\} \right].
$$

This proves that the output belongs to the permitted label set. It does not prove that the detected language is correct.

Quotation extraction permits a stronger deterministic validator:

$$
v_{\mathrm{quote}}(x,y) = \mathbf 1 \left[ y\text{ is a contiguous substring of }x \right].
$$

This ensures that the quotation is present in the source, although it does not necessarily establish that the quotation supports the associated classification.

Validators can operate at several levels:

- **Structural validation:** schema, field presence, types, and parsing;
- **Domain validation:** enumerated values, numeric ranges, and length limits;
- **Grounding validation:** whether output content occurs in or is supported by a source;
- **Cross-output validation:** whether independently produced fields are mutually consistent;
- **Semantic validation:** whether the output meaningfully satisfies the task.

Whenever a condition can be evaluated exactly—such as JSON validity, numeric bounds, or exact source inclusion—a deterministic validator is preferable.

Semantic conditions may require a human evaluator or a model-based judge. If an AI-based validator is used, it should be treated as another probabilistic estimator whose own error rate, latency, and cost must be measured. In practice, these evaluations are rarely applied to all generated outputs due to cost and latency constraints. Instead, practitioners typically sample production data for evaluation using a high-capability LLM judge or human annotators.

## 3.5. Retry and Repair Policies

When an estimator invocation fails or a validator rejects its output, the retry policy $R$ determines whether and how the architecture attempts to recover within the same primary task path.

A retry policy can be represented as:

$$
R(z,h,b)\rightarrow a_R,
$$

where:

- $z$ is the observed failure state;
- $h$ is the execution history;
- $b$ is the remaining retry, cost, or latency budget;
- $a_R$ is the next recovery action.

The action may be:

$$
a_R\in \{ \text{accept}, \text{repeat}, \text{repair}, \text{retry}, \text{invoke }\Phi, \text{stop} \}.
$$

Possible recovery actions include:

- repeating the same invocation;
- retrying with validator feedback;
- changing the prompt or inference settings;
- regenerating only the invalid field;
- repairing formatting deterministically;
- narrowing the requested output;
- stopping when the retry budget is exhausted.

Consider quotation extraction. The architecture first executes $e_{\mathrm{quote}}$ and then applies $v_{\mathrm{quote}}$. If the quotation is not found in the source, the validator can return the invalid span as structured feedback.

A repair prompt might state:

> The previous output, `<a_r>` is not an exact quotation from the source. Return a contiguous substring copied verbatim from the support ticket.

A bounded retry policy could be defined as:

$$
R_{\mathrm{quote}}(v_{\mathrm{quote}},b) = \begin{cases} \text{accept}, & v_{\mathrm{quote}}=1, \\ \text{repair and retry}, & v_{\mathrm{quote}}=0,\;b>0, \\ \text{invoke }\Phi, & v_{\mathrm{quote}}=0,\;b=0. \end{cases}
$$

Retry behavior should depend on the failure type.

For transient infrastructure failures—such as timeouts, rate limits, or interrupted connections—repeating the same operation with bounded backoff may be appropriate. Passing infrastructure error text into the model prompt is generally unnecessary and may corrupt the output.

For systematic output-validity failures, an identical retry may reproduce the same error. In these cases, validator feedback, a narrowed prompt, local repair, or a modified inference configuration is generally more effective.

A complete retry policy should specify:

- which failures are retryable;
- the maximum number of attempts;
- applicable cost and latency budgets;
- delay and backoff behavior;
- whether the estimator configuration changes;
- whether one field or the complete output is regenerated;
- whether previously valid results are preserved;
- the condition under which control passes to the fallback policy.

Retry policies affect both reliability and cost. Additional attempts can improve the probability of obtaining a valid result, but they also consume tokens, compute, and latency.

## 3.6. Fallback and Escalation Policies

A fallback policy $\Phi$ determines what happens when the primary execution path cannot produce an acceptable result within its retry or operational budget.

Unlike a retry, a fallback changes something substantial about the execution strategy. It may change:

- the estimator;
- the model;
- the method;
- the data source;
- the output contract;
- the level of automation;
- the actor responsible for the decision.

A fallback policy can be represented as:

$$
\Phi(z,h,b)\rightarrow a_{\Phi},
$$

where:

- $z$ is the unresolved failure state;
- $h$ is the execution history;
- $b$ is the remaining operational budget;
- $a_{\Phi}$ is an alternate execution or terminal action.

The action space may include:

$$
a_{\Phi}\in \{ \text{alternate estimator}, \text{alternate method}, \text{partial response}, \text{abstention}, \text{human escalation} \}.
$$

Common fallback patterns include:

- **Model escalation:** route the task from a fast, inexpensive model to a larger and more capable model;
- **Method fallback:** replace a generative estimator with a different execution method, such as an extractive span selector;
- **Graceful degradation:** return the valid portions of the output while clearly marking unavailable fields;
- **Abstention:** report that the system cannot produce a sufficiently reliable result;
- **Human escalation:** transfer the unresolved case to a human reviewer.

For example, if quotation extraction repeatedly produces unsupported text, the system might fall back from a generative estimator to an extractive method that can only select spans present in the source. If the extractive method also fails, the system might return the classifications without a supporting quotation, provided the feature contract allows partial output. For a high-risk ticket, the same failure might instead trigger human review.

Fallback behavior must therefore reflect the consequences of failure. A missing optional explanation and an incorrect high-risk classification should not necessarily invoke the same policy.

The distinction between recovery mechanisms can be summarized as follows:

| Mechanism        | Central question                                         | Example                                           |
|:-----------------|:---------------------------------------------------------|:--------------------------------------------------|
| Validation       | Is the output acceptable?                                | Check whether the quotation appears in the source |
| Retry            | Can the primary path succeed on another attempt?         | Repeat the extraction call                        |
| Repair           | Can the existing output or request be corrected locally? | Retry with the invalid span identified            |
| Fallback         | Should the system use a different method?                | Switch to extractive span selection               |
| Degradation      | Can a reduced result still provide value?                | Return the classification without a quotation     |
| Abstention       | Should the system decline to answer?                     | Mark the result as unresolved                     |
| Human escalation | Should responsibility transfer to a person?              | Send the ticket to a reviewer                     |

## 3.7. Architectural Patterns

Different choices of $P$, $E$, $G$, $V$, $R$, and $\Phi$ produce recognizable architectural patterns.

#### Single Multi-Task Architecture

A single estimator executes the complete task set:

$$
e_{\mathrm{all}}: x\mapsto (y_1,\ldots,y_n).
$$

This pattern shares context and reduces request overhead. It may lower direct token or compute cost, but it couples task failures and may create interference between instructions or output requirements. If one field fails validation, the architecture may need to regenerate the complete output.

#### Modular Architecture

Each task receives a dedicated estimator:

$$
e_1(x),e_2(x),\ldots,e_n(x).
$$

This pattern isolates evaluation and enables task-specific estimators, validators, and retries. Independent tasks can often execute concurrently. Its disadvantages include duplicated context processing, additional orchestration, and potentially higher direct inference cost.

A modular architecture may still route several estimators to the same underlying model. Modularity concerns the separation of task execution and recovery, not necessarily model diversity.

#### Hybrid Architecture

A hybrid architecture groups tasks selectively:

$$
P= \left\{ \{t_1\}, \{t_2,t_3\}, \{t_4\} \right\}.
$$

Tasks are grouped when they benefit from shared context or reasoning and separated when they require different models, validators, reliability thresholds, or recovery behavior.

Hybrid architectures are common in production because they allow engineers to balance shared execution against failure isolation.

#### Adaptive Architecture

An adaptive architecture selects execution paths dynamically. Its decisions may depend on:

- input complexity;
- model confidence;
- validation results;
- latency requirements;
- customer tier;
- safety risk;
- remaining cost budget.

For example, a lightweight estimator may process simple inputs, while ambiguous inputs are routed directly to a stronger model. Alternatively, the architecture may begin with an inexpensive estimator and escalate only if validation fails:

$$
e_{\mathrm{small}} \rightarrow \begin{cases} \text{accept}, & v(Y)=1, \\ e_{\mathrm{large}}, & v(Y)=0. \end{cases}
$$

Adaptation can occur at the level of one task group rather than replacing the entire architecture.

## 3.8. From Architecture to Reliability

The six architectural elements answer six distinct questions:

1.  $P$: Which tasks should be executed together?
2.  $E$: Which estimator should execute each task group?
3.  $G$: In what order should components run, and what depends on what?
4.  $V$: How does the system determine whether an output is acceptable?
5.  $R$: How should recoverable failures be retried or repaired?
6.  $\Phi$: What should happen when the primary execution path cannot recover?

Defining an architecture explains how outputs are produced, checked, and recovered, but it does not yet establish how reliable the resulting system is. Reliability must be evaluated at several levels: the success of an individual estimator attempt, the success of each task after recovery, the joint success of the complete feature, and the probability that the system satisfies its operational service requirements.

The next section formalizes these distinctions.

# 4. Reliability: From Estimator Attempts to System Success

Once an architecture $A=(P,E,G,V,R,\Phi)$ has been defined, its reliability must be evaluated at several distinct levels. The probability that an estimator succeeds on its first attempt is not the same as the probability that a task succeeds after validation and recovery. Similarly, the success of individual tasks does not guarantee that the complete feature—or the service experienced by the user—will succeed.

We therefore distinguish four levels of reliability:

1.  **Estimator-attempt success:** whether one estimator invocation produces a valid output;
2.  **Task-level success:** whether a task is completed successfully after architectural recovery mechanisms are applied;
3.  **Feature-level success:** whether the required combination of tasks succeeds;
4.  **Service-level success:** whether the architecture delivers an acceptable result within its operational requirements.

This hierarchy allows us to evaluate both the intrinsic performance of estimators and the additional reliability created by the architecture surrounding them.

## 4.1. Estimator-Attempt Success

Section 2 defined the success probability of estimator $e_i$ on task $t_i$ as:

$$
p_i(e_i) = \mathbb E_{x\sim\mathcal D_i} \left[ \Pr\left( S_i(x,Y_i)=1 \mid x,e_i \right) \right],
$$

where:

$$
Y_i\sim P_{e_i}(\cdot\mid x;\theta_i).
$$

This probability describes the performance of one estimator invocation before architectural recovery mechanisms are applied. It answers the question:

> If this estimator is executed once on a production input, what is the probability that its output satisfies every required task constraint?

To make the distinction explicit, we denote first-attempt success as:

$$
p_i^{(0)} = \mathbb E_{x\sim\mathcal D_i} \left[ \Pr\left( S_i(x,Y_i^{(0)})=1 \mid x,e_i \right) \right],
$$

where $Y_i^{(0)}$ is the estimator’s initial output.

First-attempt success is useful for comparing estimators because it isolates their direct behavior. However, it does not fully represent the reliability exposed by a production system. A failed initial output may be detected by a validator and corrected through repair, retry, or fallback.

## 4.2. Task-Level Architectural Success

Let $Y_{A,i}$ be the final output produced for task $t_i$ by architecture $A$, after applying any relevant validators, retries, repairs, or fallback actions.

The architecture-level probability of success for task $t_i$ is:

$$
p_i(A) = \mathbb E_{x\sim\mathcal D_i} \left[ \Pr\left( S_i(x,Y_{A,i})=1 \mid x,A \right) \right].
$$

Unlike $p_i^{(0)}$, this probability includes the complete recovery path for the task.

For example, suppose a quotation estimator succeeds on 90% of first attempts:

$$
p_{\mathrm{quote}}^{(0)}=0.90.
$$

A deterministic validator detects unsupported quotations, and a targeted retry repairs half of the detected failures. The architecture may then expose a higher final task-success rate:

$$
p_{\mathrm{quote}}(A)>p_{\mathrm{quote}}^{(0)}.
$$

The difference:

$$
\Delta p_i^{\mathrm{recovery}} = p_i(A)-p_i^{(0)}
$$

measures the reliability gained through architectural recovery.

This gain is not free. Validation, retries, and fallbacks consume additional latency, compute, tokens, and operational complexity. A later section will incorporate these costs into the total architecture objective.

## 4.3. Task-Group Success

The partition $P$ may assign several tasks to one estimator group $B_g$. Let:

$$
B_g=\{t_i,\ldots,t_j\}.
$$

The group succeeds only when every required task in the group succeeds:

$$
S_{B_g}(x,Y_{A,g}) = \prod_{t_i\in B_g} S_i(x_i,Y_{A,g,i}),
$$

where $Y_{A,g,i}$ is the portion of the group output associated with task $t_i$.

The architecture-level success probability of the group is:

$$
p_{B_g}(A) = \mathbb E_{x\sim\mathcal D_{B_g}} \left[ \Pr\left( S_{B_g}(x,Y_{A,g})=1 \mid x,A \right) \right].
$$

This quantity is especially important for multi-task estimators. A grouped estimator may return a valid sentiment label but an invalid urgency value. If both tasks are required, the complete group has failed even though one component succeeded.

Group-level evaluation should therefore preserve per-task results. Recording only a single group-level pass-or-fail metric would hide which task or constraint caused the failure.

## 4.4. Feature-Level Success

A product feature generally succeeds only when a specified combination of tasks succeeds.

If every task is mandatory, the feature-success event is:

$$
S_F(A,x) = \prod_{i=1}^{n}S_i(x_i,Y_{A,i}).
$$

The corresponding probability is:

$$
p_F(A) = \Pr\left( \bigcap_{i=1}^{n} \{S_i=1\} \mid A \right).
$$

For example, an ambiguous or sarcastic support ticket may cause both sentiment and urgency classification to fail. A malformed upstream transformation may corrupt the inputs of several downstream estimators.

If all task-success events are independent—meaning task failures are decorrelated—then the overall probability reduces to the product of each task’s success probability:

$$
p_F(A)=\prod_{i=1}^{n}p_i(A).
$$

Without assumptions about dependence, the joint feature-success probability is bounded by:

$$
\max\left( 0, \sum_{i=1}^{n}p_i(A)-(n-1) \right) \leq p_F(A) \leq \min_i p_i(A).
$$

These bounds illustrate why strong marginal task metrics do not guarantee strong feature reliability. A system may achieve 98% success on every task individually while producing a meaningfully lower probability that all tasks succeed simultaneously.

## 4.5. Service-Level Success

A semantically correct feature output may still fail operationally. A result that arrives too late, exceeds its cost budget, violates a safety requirement, or requires an unacceptable amount of manual intervention may not constitute a successful service.

Let:

- $S_F(A,x)$ denote feature-level correctness;
- $L_A(x)$ denote end-to-end latency;
- $\ell$ denote the maximum permitted latency;
- $C_A(x)$ denote execution cost;
- $b$ denote the request-level cost budget;
- $Q_A(x)$ denote compliance with safety and policy requirements.

A simplified service-success event is:

$$
S_{\mathrm{service}}(A,x) = S_F(A,x) \land \left(L_A(x)\leq\ell\right) \land \left(C_A(x)\leq b\right) \land Q_A(x).
$$

The service-level success probability is:

$$
p_{\mathrm{service}}(A) = \mathbb E_{x\sim\mathcal D} \left[ \Pr\left( S_{\mathrm{service}}(A,x)=1 \mid x,A \right) \right].
$$

This is the reliability measure that most closely reflects the user’s experience. It includes the complete execution path:

- primary estimator calls;
- deterministic transformations;
- validation;
- retry and repair;
- fallback and escalation;
- final aggregation;
- latency and policy requirements.

The distinction between feature and service success is particularly important when retries are used. Additional attempts may increase the probability of obtaining a correct output while simultaneously increasing the probability of violating the latency target.

An architecture can therefore improve:

$$
p_F(A),
$$

while reducing:

$$
p_{\mathrm{service}}(A).
$$

Reliability optimization must account for this trade-off rather than treating every increase in semantic correctness as an unconditional system improvement.

# 5. Reliability Effects of Task Coupling

## 5.1. Constraint Restrictiveness

For a fixed estimator, fixed input distribution, and fixed output distribution, adding a constraint cannot enlarge the set of valid outputs.

Let $C_a$ and $C_b$ be two constraint sets satisfying:

$$
C_a\subseteq C_b.
$$

Let $\mathcal Y_{C_a}(x)$ and $\mathcal Y_{C_b}(x)$ denote their corresponding valid output sets. Then:

$$
\mathcal Y_{C_b}(x) \subseteq \mathcal Y_{C_a}(x).
$$

For a fixed estimator $e$, it follows that:

$$
\Pr_e\left( Y\in\mathcal Y_{C_b}(x) \mid x \right) \leq \Pr_e\left( Y\in\mathcal Y_{C_a}(x) \mid x \right).
$$

Taking the expectation over the input distribution preserves the inequality:

$$
p_e(C_b)\leq p_e(C_a).
$$

This is the **monotonicity of constraints**.

The inequality is not necessarily strict. A new constraint may be redundant, or the estimator may already satisfy it for every output that passes the existing constraints. What matters is the restrictiveness of each condition, its alignment with the estimator’s capabilities, and its interaction with the other requirements.

Note that this monotonicity property holds only under fixed inference conditions; modifying the prompt or altering sampling parameters can break this inequality.

Nevertheless, it is a valuable principle to keep in mind when designing task groups. As more tasks and heterogeneous constraints are assigned to the same estimator group, the group must satisfy a larger conjunction of requirements, and its joint acceptance condition becomes more demanding.

The monotonicity principle therefore serves as an architectural warning:

> The more requirements a task group must satisfy simultaneously, the more evidence is needed to justify keeping those tasks coupled.

It encourages bounded task groups without prescribing a universally optimal group size.

## 5.2. Constraint Sensitivity and Interaction

For a fixed estimator $e$, the sensitivity to an additional constraint $c_k$ can be defined as:

$$
\Delta p_{c_k}(e) = p_e(C) - p_e(C\cup\{c_k\}).
$$

A large value indicates that the estimator frequently satisfies the original constraint set but violates the newly added condition.

The effect of a constraint may depend on the other constraints already present:

$$
\Delta p_{c_k}(e\mid C_a) \neq \Delta p_{c_k}(e\mid C_b).
$$

For example, requiring output in a given language may have little effect on an unconstrained prose task. The same language requirement may have a larger effect when combined with a fixed JSON schema, untranslated field names, exact source quotations, and strict length limits.

For two constraints $c_a$ and $c_b$, their interaction can be approximated by:

$$
I(c_a,c_b) = p(C\cup\{c_a\}) + p(C\cup\{c_b\}) - p(C) - p(C\cup\{c_a,c_b\}).
$$

A positive value indicates that the combined penalty is greater than the sum of the isolated penalties. A negative value suggests that the constraints reinforce or clarify one another.

These quantities generally cannot be derived analytically. They must be estimated on a representative evaluation dataset, and their values may differ across models, prompts, task partitions, and input segments.

Constraint sensitivity can inform task grouping. If two requirements exhibit strong negative interaction within one estimator, separating them into different groups may improve reliability. If they exhibit positive transfer, preserving the shared group may be advantageous.

## 5.3. Task Dependence, Interference, and Positive Transfer

When several tasks are assigned to one estimator group, their execution may interact.

An **interference** occurs when grouping tasks reduces their joint success relative to an appropriate modular baseline. Possible causes include:

- conflicting instructions;
- competition for limited output space;
- attention dilution;
- confusion between schemas;
- propagation of one reasoning error across several fields;
- translation of keys or quotations that should remain fixed;
- regeneration of valid fields while repairing an invalid one.

Consider an estimator instructed to translate a support-ticket summary while preserving fixed JSON keys and extracting a verbatim quotation from the original language. The translation instruction may cause it to translate the keys or alter the quotation. Additional instructions may reduce these failures, but they also increase the number of conditions the estimator must coordinate.

Grouping tasks can also create **positive transfer**. Sentiment and urgency classification may benefit from the same semantic interpretation of the ticket. Performing them together may improve consistency and avoid duplicated reasoning.

Consequently:

$$
p_F(A_{\mathrm{grouped}}) \gtrless p_F(A_{\mathrm{modular}}).
$$

The two possible scenarios are:

$$
\text{Effect} = \begin{cases} 
\text{positive transfer}, & p_F(A_{\mathrm{grouped}}) > p_F(A_{\mathrm{modular}}), \\ 
\text{interference}, & p_F(A_{\mathrm{grouped}}) < p_F(A_{\mathrm{modular}}). 
\end{cases}
$$

# 6. Total Expected Cost: Direct Execution and Failure Loss

Reliability alone does not determine whether an architecture is suitable for production. Two architectures may achieve similar feature-success rates while consuming very different amounts of compute, latency, engineering effort, and operational capacity. Conversely, the architecture with the lowest inference bill may generate failures whose downstream consequences make it substantially more expensive overall.

A complete cost model must therefore account for two categories:

1.  **Direct execution cost:** the measurable cost of running estimators, validators, retries, fallbacks, and supporting infrastructure;
2.  **Failure loss:** the economic and operational consequences of outputs that are invalid, delayed, incomplete, or otherwise unusable.

The relevant optimization target is not the cost of an isolated estimator invocation. It is the expected total cost of the complete architecture over its production input distribution.

## 6.1. The Runtime Execution Trace

An architecture does not necessarily execute the same components for every request. Routing decisions, validation outcomes, retries, and fallbacks create different execution paths.

Let:

$$
\tau_A(x,\omega)
$$

denote the runtime execution trace generated by architecture $A$ for input $x$, where $\omega$ represents the sources of randomness affecting execution. These may include:

- model sampling;
- nondeterministic model behavior;
- routing decisions;
- infrastructure failures;
- timeout behavior;
- validator outcomes;
- retry and fallback paths.

The trace records every operation performed for the request, including:

- primary estimator invocations;
- validator invocations;
- deterministic transformations;
- retries and repair attempts;
- fallback estimators;
- human escalation, when applicable.

Let:

$$
\mathcal R(\tau_A)
$$

denote the set of execution operations contained in the trace.

The direct cost of architecture $A$ on input $x$ is then:

$$
\mathcal C_{\mathrm{direct}}(A,x,\omega) = \sum_{r\in\mathcal R(\tau_A)} \mathcal C_r.
$$

This formulation ensures that retries and fallbacks are included in the direct cost. An architecture that appears inexpensive on its primary path may become costly if it frequently invokes repair prompts or escalates to larger models. Let’s see how $C_r$ can be defined.

## 6.2. Direct API Cost

For estimators accessed through an API provider, cost is usually based on the volume and type of tokens processed.

For an invocation $r$, a basic cost model is:

$$
\mathcal C_{\mathrm{API},r} = c_{\mathrm{in},r}N_{\mathrm{in},r} + c_{\mathrm{out},r}N_{\mathrm{out},r},
$$

where:

- $c_{\mathrm{in},r}$ is the price per input token;
- $c_{\mathrm{out},r}$ is the price per output token;
- $N_{\mathrm{in},r}$ is the number of input tokens;
- $N_{\mathrm{out},r}$ is the number of output tokens.

The prices depend on the model used by the invocation:

$$
c_{\mathrm{in},r} = c_{\mathrm{in}}(\pi_r), \qquad c_{\mathrm{out},r} = c_{\mathrm{out}}(\pi_r).
$$

A more complete model may distinguish between uncached input, cached input, generated output, internal reasoning, and tool usage:

$$
\mathcal C_{\mathrm{API},r} = c_{\mathrm{uncached},r}N_{\mathrm{uncached},r} + c_{\mathrm{cached},r}N_{\mathrm{cached},r} + c_{\mathrm{out},r}N_{\mathrm{out},r} + \mathcal C_{\mathrm{tools},r}.
$$

The total API cost of the execution trace is:

$$
\mathcal C_{\mathrm{API}}(A,x,\omega) = \sum_{r\in\mathcal R_{\mathrm{API}}(\tau_A)} \mathcal C_{\mathrm{API},r}.
$$

This includes every model call actually made, not only the successful one. If an architecture calls a small model twice and then escalates to a larger model, all three calls contribute to the request’s direct cost.

The number of input tokens depends on more than the original user payload. It may include:

- system instructions;
- task-specific prompts;
- few-shot examples;
- retrieved context;
- intermediate outputs;
- validator feedback;
- conversation history;
- serialized tool results.

Similarly, output-token cost includes invalid outputs that are later discarded.

## 6.3. Direct Self-Hosted Cost

For self-hosted estimators, request cost is not determined by a public per-token price. It must be allocated from infrastructure and operational expenses.

A simplified per-invocation proxy is:

$$
\mathcal C_{\mathrm{compute},r} = u(h_r)\,d_r,
$$

where:

- $u(h_r)$ is the allocated cost per unit of device time for hardware tier $h_r$;
- $d_r$ is the amount of device time allocated to invocation $r$.

To refine the formula, we can also include:

- hardware utilization;
- batching efficiency;
- model memory requirements;
- idle capacity;
- autoscaling behavior;
- queueing;
- storage and networking;
- redundancy requirements;
- deployment and maintenance labor.

A broader self-hosted model is:

$$
\mathcal C_{\mathrm{host}}(A,x,\omega) = \mathcal C_{\mathrm{compute}} + \mathcal C_{\mathrm{capacity}} + \mathcal C_{\mathrm{storage}} + \mathcal C_{\mathrm{network}} + \mathcal C_{\mathrm{operations}},
$$

where shared expenses must be allocated across the requests served during the relevant period.

## 6.4. Latency as an Architectural Quantity

In our framework, we define latency as an operational metric:

$$
L_A(x,\omega).
$$

For a sequential pipeline, end-to-end latency may approximately equal the sum of component latencies. For an architecture with concurrent branches, it is determined primarily by the critical execution path:

$$
L_A(x,\omega) \approx \max_{\gamma\in\Gamma_A} \sum_{r\in\gamma}L_r,
$$

where $\Gamma_A$ is the set of execution paths through the runtime graph.

This distinction matters when comparing grouped and modular architectures. Three independent estimators executed concurrently may consume more total compute while achieving lower wall-clock latency than one long multi-task generation.

Latency can be handled in two ways.

First, it can be imposed as a service constraint:

$$
\Pr\left( L_A(x,\omega)\leq\ell \right) \geq q,
$$

where $\ell$ is the latency target and $q$ is the required proportion of requests meeting it.

Second, latency can be translated into economic loss when slow responses cause abandonment, service-level penalties, reduced conversion, or downstream delay. In that case, its consequences belong in the failure-loss function introduced below.

## 6.5. Direct-Cost Effects of Task Grouping

Task grouping often creates a direct-cost advantage because grouped estimators can share input context and request overhead.

Suppose $t_i,\ldots,t_j$ all operate on the same source document. A modular architecture may repeatedly send that document to separate estimators:

$$
\sum_{k=i}^{j}N_{\mathrm{in}}(e_k).
$$

A grouped estimator may send it only once:

$$
N_{\mathrm{in}}(e_{i:j}) < \sum_{k=i}^{j}N_{\mathrm{in}}(e_k).
$$

Grouped execution may also reduce repeated instructions, JSON wrappers, network requests, and intermediate serialization.

However, the direct-cost advantage is not universal. A grouped architecture may require:

- a larger and more expensive model;
- a longer prompt coordinating heterogeneous tasks;
- additional output explaining several decisions;
- regeneration of every field when one field fails;
- more expensive validation;
- a longer context window;
- reduced opportunities for parallelism.

A modular architecture may duplicate context, but it can also:

- assign inexpensive specialized models to simple tasks;
- execute independent tasks concurrently;
- retry only the failed task;
- avoid regenerating already-valid outputs;
- omit estimators for tasks that are not required for a particular input.

Prompt caching can further reduce repeated input cost in modular architectures. In self-hosted systems, reuse of the key-value cache may similarly reduce repeated context computation.

Task grouping should therefore be described as having a potential direct-cost advantage under comparable model and execution assumptions, not as being universally cheaper.

## 6.6. Failure Loss

Direct execution cost captures what the system spends while running. It does not capture the full consequence of delivering an unacceptable result.

Let:

$$
\Lambda(A,x,Y_A,\tau_A)\geq 0
$$

denote the loss produced by the architecture’s final outcome, where:

- $Y_A$ is the final output;
- $\tau_A$ is the complete execution trace.

This loss may contain several components:

$$
\Lambda = \Lambda_{\mathrm{remediation}} + \Lambda_{\mathrm{delay}} + \Lambda_{\mathrm{customer}} + \Lambda_{\mathrm{business}} + \Lambda_{\mathrm{risk}}.
$$

These terms may represent:

- **Remediation loss:** human review, manual correction, incident response, or engineering investigation;
- **Delay loss:** missed deadlines, blocked downstream workflows, or service-level violations;
- **Customer loss:** reduced trust, abandonment, support requests, or churn;
- **Business loss:** incorrect decisions, missed opportunities, or corrupted downstream data;
- **Risk loss:** legal, safety, compliance, or reputational exposure.

These costs are called **hidden costs** because they do not appear directly on an API invoice or infrastructure dashboard and may be difficult to quantify. Nevertheless, they are real consequences of architectural behavior.

The same technical failure can have different losses depending on context. An incorrect sentiment label on a low-priority internal dashboard may have little consequence. An incorrect urgency classification that delays intervention for a critical customer may be substantially more expensive.

Therefore, the loss function should depend on:

- which task failed;
- the input on which it failed;
- whether the failure was detected;
- whether an invalid output reached the user;
- whether partial output remained useful;
- how long recovery took;
- whether human intervention was required;
- the business importance of the affected request.

## 6.7. Total Expected Architecture Cost

The total cost of an architecture for one execution is:

$$
\mathcal T(A,x,\omega) = \mathcal C_{\mathrm{direct}}(A,x,\omega) + \Lambda(A,x,Y_A,\tau_A).
$$

The expected total cost over the production input distribution and runtime randomness is:

$$
\mathbb E[\mathcal T(A)] = \mathbb E_{\substack{x\sim\mathcal D\\\omega}} \left[ \mathcal C_{\mathrm{direct}}(A,x,\omega) + \Lambda(A,x,Y_A,\tau_A) \right].
$$

For brevity, we denote this quantity as:

$$
\mathcal T(A) = \mathbb E_{\substack{x\sim\mathcal D\\\omega}} \left[ \mathcal C_{\mathrm{direct}} + \Lambda \right].
$$

This formulation incorporates:

- the primary execution path;
- conditional routing;
- validation;
- retries;
- repair operations;
- fallback estimators;
- partial outputs;
- abstentions;
- unresolved failures;
- latency and business consequences.

## 6.8. A Simplified Failure-Probability Approximation

When detailed failure losses are unavailable, teams may use a simplified approximation.

Suppose loss is zero when the service succeeds and has an average value $\overline{\Lambda}_{\mathrm{fail}}$ when it fails. Then:

$$
\mathcal T(A) \approx \mathbb E[\mathcal C_{\mathrm{direct}}(A)] + \left( 1-p_{\mathrm{service}}(A) \right) \overline{\Lambda}_{\mathrm{fail}},
$$

where:

$$
\overline{\Lambda}_{\mathrm{fail}} = \mathbb E[ \Lambda \mid S_{\mathrm{service}}=0 ].
$$

This approximation is useful for initial architectural comparisons, but it hides differences between failure types. A more informative model separates failures into categories:

$$
\mathcal T(A) \approx \mathbb E[\mathcal C_{\mathrm{direct}}(A)] + \sum_{z\in\mathcal Z} \Pr(z\mid A)\, \overline{\Lambda}_z,
$$

where:

- $\mathcal Z$ is the set of relevant failure modes;
- $\Pr(z\mid A)$ is the probability of failure mode $z$;
- $\overline{\Lambda}_z$ is its average consequence.

For the support-ticket feature, $\mathcal Z$ might include:

- incorrect language;
- incorrect sentiment;
- missed urgency;
- false urgency;
- unsupported quotation;
- schema failure;
- latency violation;
- unresolved abstention.

A missed urgent ticket is likely to have a different loss from a missing optional quotation. Modeling those outcomes separately produces a more useful architecture comparison.

## 6.9. Comparing Grouped and Modular Architectures

Consider two candidate architectures:

- $A_{\mathrm{grouped}}$: one estimator executes several tasks together;
- $A_{\mathrm{modular}}$: the tasks are executed by separate estimators.

Suppose the grouped architecture has lower expected direct cost:

$$
\mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{grouped}}) ] < \mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{modular}}) ].
$$

This inequality alone is not sufficient to select it.

The grouped architecture is preferable only when:

$$
\mathcal T(A_{\mathrm{grouped}}) < \mathcal T(A_{\mathrm{modular}}).
$$

Expanding both sides:

$$
\mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{grouped}}) ] + \mathbb E[ \Lambda(A_{\mathrm{grouped}}) ] < \mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{modular}}) ] + \mathbb E[ \Lambda(A_{\mathrm{modular}}) ].
$$

Rearranging gives a useful decision rule:

$$
\underbrace{ \mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{modular}}) ] - \mathbb E[ \mathcal C_{\mathrm{direct}}(A_{\mathrm{grouped}}) ] }_{\text{direct savings from grouping}} > \underbrace{ \mathbb E[ \Lambda(A_{\mathrm{grouped}}) ] - \mathbb E[ \Lambda(A_{\mathrm{modular}}) ] }_{\text{additional failure loss from grouping}}.
$$

In plain language:

> Grouping is economically justified when its direct execution savings exceed any additional loss created by coupled failures, broader retries, reduced specialization, or lower reliability.

If grouping also improves reliability through positive transfer, both sides may favor the grouped architecture. If modular execution enables cheaper specialized estimators or highly localized retries, the modular architecture may have lower direct cost as well as lower failure loss.

The comparison is empirical. It cannot be resolved from task count or token count alone.

## 6.10. The Cost of Partial Failure and Recovery Scope

The scope of recovery strongly influences total cost.

Suppose a grouped estimator returns four fields and only the quotation fails validation. If the architecture must regenerate the complete response, its retry cost is approximately:

$$
\mathcal C_{\mathrm{retry}}^{\mathrm{grouped}} = \mathcal C(e_{\mathrm{all}}).
$$

In a modular architecture, only the quotation estimator may need to be repeated:

$$
\mathcal C_{\mathrm{retry}}^{\mathrm{modular}} = \mathcal C(e_{\mathrm{quote}}).
$$

The modular architecture may therefore have a higher successful-path cost but a lower recovery cost.

This distinction becomes increasingly important when:

- failures are frequent;
- outputs are long;
- one task is substantially less reliable than the others;
- task-specific fallback models are available;
- successful fields can be preserved;
- the consequences of regenerating correct fields are significant.

Architectural comparisons should therefore report at least:

- cost when the primary path succeeds;
- expected retry and repair cost;
- expected fallback cost;
- cost by failure mode;
- cost per successful feature response.

A useful aggregate metric is:

$$
\mathcal C_{\mathrm{successful\ outcome}}(A) = \frac{ \mathbb E[ \mathcal C_{\mathrm{direct}}(A)+\Lambda(A) ] }{ p_F(A) },
$$

provided that feature success is defined consistently across the architectures being compared. This ratio estimates total cost per successful outcome over a large population of requests.

# 7. Optimizing Estimators and Architectures

The preceding sections defined an architecture as:

$$
A=(P,E,G,V,R,\Phi)
$$

and its total expected cost as:

$$
\mathcal T(A) = \mathbb E_{\substack{x\sim\mathcal D\\\omega}} \left[ \mathcal C_{\mathrm{direct}}(A,x,\omega) + \Lambda(A,x,Y_A,\tau_A) \right].
$$

The objective of optimization is to find an architecture that reduces this total expected cost while continuing to satisfy the feature’s reliability, latency, safety, and product requirements.

Let $\mathcal A$ denote the set of candidate architectures. The global optimization problem is:

$$
A^* = \arg\min_{A\in\mathcal A} \mathcal T(A)
$$

subject to given constraints such as:

$$
p_F(A)\geq\tau_F, \quad \Pr(L_A\leq\ell)\geq q, \quad Q_A=1.
$$

Here:

- $\tau_i$ is the minimum acceptable reliability for task $t_i$;
- $\tau_F$ is the minimum acceptable feature-level reliability;
- $\ell$ is the latency target;
- $q$ is the required proportion of requests meeting that target;
- $Q_A$ represents mandatory safety or policy compliance.

This constrained formulation prevents the optimizer from lowering cost by simply accepting lower-quality outputs. An architecture that violates a critical requirement is not an inexpensive solution; it is an infeasible one.

## 7.1. Two Levels of Optimization

Optimization can occur at two related levels:

1.  **Estimator optimization:** improve the component assigned to a fixed architectural role;
2.  **Architecture optimization:** change how tasks, estimators, validators, and recovery mechanisms are composed.

Suppose estimator $e_g$ is assigned to task group $B_g$. Holding the rest of the architecture fixed, the locally optimal estimator is:

$$
e_g^* = \arg\min_{e\in\mathcal E(B_g)} \mathcal T\left( A[e_g\leftarrow e] \right),
$$

subject to the applicable reliability and service constraints.

The notation:

$$
A[e_g\leftarrow e]
$$

means that estimator $e_g$ is replaced by candidate estimator $e$, while the remainder of the architecture remains unchanged.

This is a local optimization. It asks:

> Given the current task partition, execution graph, validators, retries, and fallbacks, which estimator should occupy this role?

Architecture optimization asks a broader question:

> Should this role exist in its current form at all?

The globally optimal design may require changing the task group $B_g$, moving a constraint into deterministic code, executing tasks in parallel, introducing a validator, modifying the retry policy, or eliminating the estimator entirely.

A locally optimal estimator can therefore belong to a globally suboptimal architecture.

## 7.2. The Golden Estimator Is Conditional

For a model-based estimator:

$$
e_g=e(\cdot;\pi_g,\rho_g,s_g),
$$

where:

- $\pi_g$ is the model;
- $\rho_g$ is the prompt and context-construction strategy;
- $s_g$ contains the inference settings.

The estimator-level optimization problem is:

$$
(\pi_g^*,\rho_g^*,s_g^*) = \arg\min_{\pi,\rho,s} \mathcal T \left( A[ e_g\leftarrow e(\cdot;\pi,\rho,s) ] \right).
$$

The resulting configuration can be called the **golden estimator** for that architectural role.

However, the term must be interpreted conditionally. An estimator is not universally optimal. It is optimal only relative to:

- a particular task group;
- a particular architecture;
- a particular input distribution;
- a particular cost model;
- a particular set of service requirements;
- a particular set of available models and infrastructure.

If any of these conditions changes, the golden estimator may change as well.

## 7.3. Estimator-Level Optimization Levers

Estimator optimization modifies how an existing task group is executed without initially changing the larger architecture.

#### Model Selection

The first lever is selecting the model $\pi_g$.

Not every task requires a frontier model. A lightweight classifier, encoder, or Small Language Model may achieve the required reliability at substantially lower cost.

For a fixed task group $B_g$, the objective is:

$$
\pi_g^* = \arg\min_{\pi\in\Pi_g} \mathcal T \left( A[ e_g\leftarrow e(\cdot;\pi,\rho_g,s_g) ] \right),
$$

where $\Pi_g$ is the set of candidate models for the role.

Model selection should consider:

- direct token or compute cost;
- first-attempt success;
- output length;
- latency;
- validator rejection;
- retry frequency;
- fallback frequency;
- failure type and consequence.

A cheaper model is not an improvement if it causes enough retries or escalations to increase total expected cost.

#### Prompt and Context Optimization

The prompt $\rho_g$ includes more than written instructions. It may contain:

- system instructions;
- output schemas;
- demonstrations;
- retrieved context;
- tool descriptions;
- intermediate results;
- validator feedback;
- rules for abstention.

Prompt optimization seeks:

$$
\rho_g^* = \arg\min_{\rho} \mathcal T \left( A[ e_g\leftarrow e(\cdot;\pi_g,\rho,s_g) ] \right).
$$

Prompt changes can reduce cost through several mechanisms:

- improving first-attempt success;
- reducing input or output length;
- preventing unnecessary explanations;
- improving schema adherence;
- reducing validator rejection;
- reducing retry and fallback frequency.

However, longer prompts may increase direct cost even when they improve reliability. Prompt optimization should therefore measure total architecture cost rather than accuracy alone.

Common prompt-optimization methods include:

- manual error analysis and revision;
- dynamic few-shot selection;
- retrieval of examples similar to the current input;
- iterative prompt mutation;
- evaluator-guided search;
- Pareto optimization across cost, latency, and reliability.

Automated prompt optimization remains vulnerable to overfitting. Prompt variants should be evaluated on held-out production-like data rather than only on the examples used to generate them.

#### Inference-Setting Optimization

Inference settings $s_g$ may include:

- temperature;
- top-$p$;
- maximum output length;
- stopping conditions;
- reasoning budget;
- tool-choice settings;
- decoding strategy.

The objective is:

$$
s_g^* = \arg\min_s \mathcal T \left( A[ e_g\leftarrow e(\cdot;\pi_g,\rho_g,s) ] \right).
$$

Inference settings are often inexpensive to experiment with because they do not require model training. Grid search, Bayesian optimization, or other numerical search methods can be applied when the parameter space is manageable.

#### Model Adaptation

When routing, prompting, and inference settings are insufficient, the underlying model may be adapted through:

- supervised fine-tuning;
- parameter-efficient methods such as LoRA;
- preference optimization;
- distillation;
- task-specific representation learning.

The optimization problem becomes:

$$
\pi_g' = \arg\min_{\pi'} \mathcal T \left( A[ e_g\leftarrow e(\cdot;\pi',\rho_g,s_g) ] \right).
$$

Model adaptation has higher experimentation and maintenance cost than prompt or inference optimization. Its training expense should be amortized across expected production volume:

$$
\mathcal C_{\mathrm{adapted}} = \mathcal C_{\mathrm{inference}} + \frac{ \mathcal C_{\mathrm{training}} + \mathcal C_{\mathrm{evaluation}} + \mathcal C_{\mathrm{maintenance}} }{ N_{\mathrm{expected\ requests}} }.
$$

Distillation may reduce the long-term cost floor by transferring behavior from a large teacher model to a smaller student model. However, the assumption that reliability remains stable must be demonstrated through evaluation, especially on rare or high-loss inputs.

## 7.4. Architecture-Level Optimization Levers

Optimizing architecture can be done by affecting one or several of its components. From section 3, architectures are defined as:

$$
A=(P,E,G,V,R,\Phi).
$$

#### Optimizing $P$: Task Partitioning

Changing $P$ alters which tasks share an estimator.

Possible experiments include:

- splitting an unreliable task from a large multi-task group;
- grouping semantically related classification tasks;
- isolating a task with expensive failure consequences;
- separating tasks with incompatible output requirements;
- moving formatting into deterministic code.

For example:

$$
P_1= \left\{ \{t_1,t_2,t_3,t_4\} \right\}
$$

may be compared with:

$$
P_2= \left\{ \{t_1\}, \{t_2,t_3\}, \{t_4\} \right\}.
$$

The comparison should measure:

- direct cost;
- joint feature success;
- per-task failure;
- retry scope;
- fallback frequency;
- latency;
- total expected cost.

The goal is not to maximize or minimize the number of groups. It is to find task boundaries that preserve useful shared execution while avoiding unnecessary coupling.

#### Optimizing $E$: Estimator Assignment and Routing

Changing $E$ assigns different execution mechanisms to task groups.

A fixed assignment uses one estimator for every input in a group:

$$
E(B_g)=e_g.
$$

An adaptive assignment uses a routing policy:

$$
E(B_g,x)=e_{g,r(x)},
$$

where $r(x)$ selects an estimator based on properties of the input.

For example:

$$
r(x)= \begin{cases} \text{small model}, & x\in\mathcal D_{\mathrm{simple}},\\ \text{large model}, & x\in\mathcal D_{\mathrm{complex}}. \end{cases}
$$

Routing decisions may use properties of the input - such as its length, language, and topic — alongside estimates of complexity and classifier confidence. Business context, including customer tier and risk category, may also influence routing, as can similarity to known failure cases.

Routing reduces cost when inexpensive estimators handle easy inputs reliably while expensive estimators are reserved for inputs that need them. However, routing introduces another decision component whose errors must be evaluated.

#### Optimizing $G$: Execution Graph

Changing $G$ modifies dependency structure and concurrency.

Possible optimizations include:

- executing independent estimators in parallel;
- delaying expensive tasks until cheaper prerequisite checks pass;
- avoiding downstream calls when an upstream result makes them unnecessary;
- passing only relevant intermediate context;
- preserving completed branches during local retries;
- reordering validators to fail cheaply and early.

For example, a low-cost eligibility check may prevent an expensive generation call:

$$
v_{\mathrm{eligibility}}(x) \rightarrow \begin{cases} e_{\mathrm{expensive}}(x), & v=1,\\ \text{stop}, & v=0. \end{cases}
$$

Graph optimization can reduce both direct cost and latency without changing estimator quality.

#### Optimizing $V$: Validation

Validator optimization involves deciding:

- which outputs require validation;
- which conditions can be checked deterministically;
- when model-based evaluation is justified;
- how validation thresholds are calibrated;
- what feedback validators return;
- where validators occur in the graph.

A validator is valuable when the expected loss it prevents exceeds its cost:

$$
\mathbb E[ \Lambda_{\mathrm{without}\ v} ] - \mathbb E[ \Lambda_{\mathrm{with}\ v} ] > \mathbb E[ \mathcal C_v ].
$$

This comparison should include false acceptances, false rejections, additional latency, and unnecessary recovery actions.

#### Optimizing $R$: Retry and Repair

Retry optimization determines:

- which failures are retryable;
- how many attempts are allowed;
- whether the estimator configuration changes;
- whether validator feedback is supplied;
- whether valid fields are preserved;
- when retrying should stop.

The optimal retry count is not necessarily the count that maximizes eventual success. Each additional retry has diminishing value when failures are correlated and increasing cost when latency or token budgets are tight.

A retry should be executed when its expected benefit exceeds its incremental cost:

$$
q_r\Delta\Lambda_r > \mathcal C_r+\Delta\Lambda_{\mathrm{delay},r},
$$

where:

- $q_r$ is the estimated probability that retry $r$ succeeds;
- $\Delta\Lambda_r$ is the failure loss avoided by success;
- $\mathcal C_r$ is the direct retry cost;
- $\Delta\Lambda_{\mathrm{delay},r}$ is the additional loss caused by delay.

This policy can vary by failure type and input segment.

#### Optimizing $\Phi$: Fallback and Escalation

Fallback optimization determines when the architecture should:

- switch models;
- switch methods;
- degrade gracefully;
- abstain;
- escalate to a human.

A larger fallback model should not be invoked merely because it is more capable. It should be invoked when its expected reduction in failure loss exceeds its additional cost and latency.

For fallback action $a_{\Phi}$:

$$
\mathbb E[ \Lambda\mid\text{no fallback} ] - \mathbb E[ \Lambda\mid a_{\Phi} ] > \mathbb E[ \mathcal C(a_{\Phi}) ].
$$

High-loss tasks may justify aggressive escalation. Low-value optional tasks may justify abstention or partial output instead.

## 7.5. Estimating the Optimization Objective

The true production distribution $\mathcal D$ and true failure-loss function $\Lambda$ are rarely known. Optimization must rely on empirical estimates.

Given an evaluation dataset:

$$
D_{\mathrm{eval}} = \{x_1,\ldots,x_N\},
$$

the empirical total cost is:

$$
\widehat{\mathcal T}(A) = \frac{1}{N} \sum_{n=1}^{N} \left[ \mathcal C_{\mathrm{direct}}(A,x_n) + \widehat{\Lambda}(A,x_n) \right].
$$

The dataset should represent:

- common production inputs;
- rare but expensive failure cases;
- different languages and input lengths;
- ambiguous examples;
- structurally unusual inputs;
- high-risk customer or business segments;
- known historical failures.

Uniform sampling from production traffic may underrepresent rare high-loss events. Stratified sampling or importance weighting may therefore be necessary.

If example $x_n$ receives weight $w_n$, the objective becomes:

$$
\widehat{\mathcal T}_w(A) = \frac{ \sum_{n=1}^{N} w_n \left[ \mathcal C_{\mathrm{direct}}(A,x_n) + \widehat{\Lambda}(A,x_n) \right] }{ \sum_{n=1}^{N}w_n }.
$$

Optimization should use separate datasets for:

- development;
- architecture and prompt selection;
- final held-out evaluation.

Repeatedly optimizing against the same dataset risks overfitting the architecture to its visible examples.

## 7.6. When Costs Cannot Be Reduced to Money

Not every reliability, latency, safety, or product requirement can be credibly converted into a monetary value.

In such cases, optimization should remain constrained or multi-objective rather than forcing every outcome into an arbitrary currency value.

The architecture may be evaluated as a vector:

$$
J(A) = \left( \mathbb E[\mathcal C_{\mathrm{direct}}], 1-p_F(A), L_{q}(A), \mathbb E[\Lambda_{\mathrm{human}}], \ldots \right).
$$

The goal is then to identify architectures on the Pareto frontier: architectures for which no objective can be improved without worsening another.

Product and engineering stakeholders can select among these candidates using explicit service thresholds and risk tolerance. This is often more honest than pretending that every consequence has a precisely known monetary value.

## 7.7. A Safe Experimental Progression

Architecture optimization should move through increasingly realistic evaluation stages.

#### Offline Evaluation

Run candidate estimators and architectures against a fixed evaluation set. Measure:

- task and feature success;
- failure types;
- token or compute cost;
- latency;
- retries and fallbacks;
- estimated total cost.

Offline evaluation is inexpensive and repeatable but cannot fully reproduce production behavior.

#### Replay Evaluation

Execute candidates on recorded production inputs when privacy, consent, and retention rules permit. This provides a more representative estimate of the true input distribution without affecting users.

#### Shadow Evaluation

Run the candidate architecture alongside the production architecture without exposing its outputs to users. Shadow evaluation reveals operational characteristics such as real latency, routing behavior, and fallback frequency.

#### Canary Deployment

Expose the candidate to a small, controlled fraction of traffic. Monitor service metrics and define automatic rollback conditions.

#### Controlled Expansion

Increase traffic only after the candidate satisfies reliability, safety, latency, and cost requirements over a sufficiently representative sample.

This progression prevents an apparent offline improvement from becoming an uncontrolled production regression.

# 8. Recommended Optimization Flow

A practical optimization process can follow eleven stages.

## 8.1. Define the Feature Contract

Specify:

- required tasks;
- acceptance conditions;
- optional outputs;
- reliability targets;
- latency targets;
- safety constraints;
- consequences of each failure type.

Optimization is impossible without a stable definition of success.

## 8.2. Establish a Capable Baseline

Begin with an architecture that is simple to understand and sufficiently capable to deliver acceptable reliability.

A general-purpose frontier model may be appropriate for the initial baseline, particularly when little production data is available. However, the rollout should remain limited, observable, and protected by validation and fallback mechanisms. A capable model does not eliminate the need for production safeguards.

## 8.3. Instrument the Complete Execution Trace

Record:

- estimator configuration;
- task-group assignment;
- validator results;
- retry and repair actions;
- fallback decisions;
- token or compute consumption;
- latency;
- final task and feature outcomes.

Without path-level telemetry, engineers cannot determine which architectural component is responsible for cost or failure.

## 8.4. Collect Representative Data

Gather real inputs, failure examples, corrections, and operational feedback. Construct an evaluation dataset that preserves both frequent cases and rare high-loss events.

Production data collection must follow applicable privacy, security, retention, and consent requirements.

## 8.5. Identify the Dominant Cost Drivers

Determine whether total cost is dominated by:

- primary inference;
- long outputs;
- repeated context;
- retries;
- fallback models;
- human review;
- latency violations;
- one high-loss failure mode.

Optimization should begin with the component having the highest plausible effect on $\mathcal T(A)$, not automatically with the component that is easiest to change.

A useful prioritization heuristic is:

$$
\text{Priority} \approx \frac{ \text{Expected reduction in }\mathcal T }{ \text{Experiment cost and risk} }.
$$

## 8.6. Optimize Low-Risk Components

Typical early experiments include:

- replacing model-enforced formatting with deterministic code;
- shortening prompts;
- reducing unnecessary output;
- tuning inference settings;
- improving validator feedback;
- preserving valid fields during retry;
- eliminating redundant calls.

These changes are often reversible and require little training data.

## 8.7. Optimize Estimator Assignment and Routing

Test smaller or specialized estimators for well-defined task groups. Introduce routing when input segments have meaningfully different difficulty or risk profiles.

Measure the complete path cost, including misrouting, retries, and escalation.

## 8.8. Revisit the Architecture

Once task-level evidence is available, reconsider:

- task partition $P$;
- estimator assignment $E$;
- execution graph $G$;
- validators $V$;
- retry policies $R$;
- fallbacks $\Phi$.

A persistent prompt problem may actually be a task-grouping problem. A costly model may be compensating for an unnecessarily broad output contract. Repeated retries may indicate that the architecture needs another method rather than another prompt revision.

## 8.9. Adapt or Distill Models When Justified

Fine-tuning and distillation become attractive when:

- the task is stable;
- evaluation data is sufficiently representative;
- request volume supports amortizing training cost;
- prompt and architecture improvements have reached diminishing returns;
- the expected cost-floor reduction is substantial.

Model adaptation should be treated as one optimization lever, not as the default final stage of every system.

## 8.10. Continuous Reoptimization

The optimal architecture is not permanent. It may change when:

- the production input distribution shifts;
- user behavior changes;
- a model provider releases a new model;
- model pricing changes;
- prompt caching becomes available;
- latency requirements change;
- task definitions evolve;
- new failure modes are discovered;
- traffic volume changes the economics of self-hosting or fine-tuning.

The optimization process should therefore be repeated periodically or triggered by significant changes in system conditions.

Let the architecture selected at time $t$ be:

$$
A_t^* = \arg\min_{A\in\mathcal A_t} \mathcal T_t(A).
$$

A later change in the candidate set, cost function, or input distribution may produce:

$$
A_{t+1}^* \neq A_t^*.
$$

Production monitoring should detect when the assumptions supporting the current architecture no longer hold.

## 8.11. From Golden Estimators to a Golden Architecture

The **golden estimator** is the best component for a defined architectural role under a particular set of conditions.

The **golden architecture** is the complete composition of task boundaries, estimators, execution dependencies, validators, retries, and fallbacks that minimizes total expected cost while satisfying the feature’s service requirements:

$$
A^* = (P^*,E^*,G^*,V^*,R^*,\Phi^*).
$$

The distinction is fundamental. Optimizing an estimator asks how to execute an existing task group more efficiently. Optimizing an architecture asks whether the system has defined and composed those task groups correctly in the first place.

The central objective of AI engineering is therefore not to find the cheapest model, the most capable model, or the best prompt in isolation. It is to design and continuously refine an architecture whose components work together to deliver reliable outcomes at the lowest sustainable total cost.

# Conclusion

Production AI engineering is often framed as a search for the right model or prompt. That framing is too narrow. A model invocation is only one execution component inside a system whose reliability and cost emerge from the interaction of many architectural decisions.

This article began by decomposing a product feature into explicitly defined operational tasks. Each task was associated with an input space, an output space, a set of validity constraints, and operational requirements. This decomposition establishes the contracts against which implementations can be evaluated.

We then introduced estimators as the mechanisms that attempt to execute those tasks. An estimator may be a language model, classifier, retrieval system, heuristic, deterministic function, or human process. Its performance is meaningful only relative to a defined task, configuration, input distribution, and success condition.

Estimators become production systems only when they are composed into an architecture:

$$
A=(P,E,G,V,R,\Phi).
$$

The architecture determines which tasks share an estimator, which execution paths are followed, how outputs are validated, what is retried, and when the system falls back, degrades, abstains, or escalates.

This architectural perspective changes how reliability should be understood. First-attempt estimator accuracy is only one layer. A complete assessment must distinguish estimator-attempt success, task success after recovery, joint feature success, and service success under latency, cost, and policy constraints.

It also changes how cost should be calculated. Direct inference cost matters, but it does not represent the full economics of a production system. Retries, validators, fallback calls, infrastructure, delayed workflows, human remediation, and failures reaching users all contribute to the true cost of an architecture.

The resulting objective is:

$$
A^* = \arg\min_{A\in\mathcal A} \mathbb E \left[ \mathcal C_{\mathrm{direct}}(A)+\Lambda(A) \right],
$$

subject to the feature’s reliability, latency, safety, and product requirements.

Within this framework, the choice between grouped and modular execution is not ideological. Grouping tasks can reduce context duplication and request overhead while creating useful shared reasoning. It can also couple failures, narrow the joint acceptance condition, and make recovery more expensive. Modularity can improve specialization, observability, and retry precision while increasing orchestration and repeated computation.

The best architecture is therefore rarely found at either extreme. It must be discovered empirically by measuring complete execution paths over representative production inputs.

Optimization consequently occurs at two levels. Estimator optimization improves a component by changing its model, prompt, context, inference settings, or training. Architecture optimization reconsiders the task partition, estimator assignments, dependency graph, validators, retries, and fallbacks. A locally optimal estimator may still belong to a globally inefficient system.

This leads to the distinction between two optimization targets:

- A **golden estimator** is the most cost-effective component for a defined architectural role.
- A **golden architecture** is the complete system that delivers the required service at the lowest sustainable total cost.

Neither remains golden forever. Input distributions shift, model capabilities improve, prices change, caching becomes available, and product requirements evolve. Production AI engineering must therefore be treated as a continuous process of measurement, experimentation, and architectural revision.

The central principle is simple:

> Do not optimize the model call in isolation. Optimize the system that turns uncertain model behavior into a reliable product outcome.
