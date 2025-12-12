Here is how I’d approach this:

1. Brief, blunt critique of the current draft.
2. A fully rewritten main text (abstract → conclusion), in LaTeX-ready form, that preserves your method and results but fixes structure, narrative, and related work.

You can drop this into your `.tex` by replacing your current `\begin{abstract}...\end{abstract}` and all sections through `\bibliography{references}`. Keep your tables, figures, and appendix as they are.

---

### 1. High‑level critique of the current draft

Based on the PDF you uploaded, your core idea is solid and fairly distinctive: instantiate Sutton’s OaK view in a FOL micro‑world with inference rules as options, a solver as knowledge, and DPO for preference alignment. 

The issues are mostly writing and narrative:

* **Introduction**

  * Spends too long rehashing standard CoT / self-consistency points.
  * Your perspective (reasoning as option selection + OaK) appears late and is under-emphasized.
  * Contributions are listed but not tightly framed around what is actually new (finite option set, explicit q̂, solver-driven DPO loop, empirical gains).

* **Related Work**

  * Reads as a catalog instead of an argument.
  * It describes prior methods but doesn’t clearly say “here is exactly what they do not give you that Sokrates does.”
  * The connection to OaK is underspecified; you say you instantiate OaK but don’t really walk the reader through that mapping.

* **Problem Setup / Method**

  * The logic world, state representation, and option semantics are only lightly specified.
  * The micro OaK loop is conceptually nice but you don’t push the story: “experience → knowledge → policy improvement”.
  * The role of q̂ is undersold; you mention calibration but don’t explain why it matters or how it could be used.

* **Experimental Narrative**

  * The headline story is strong (27.3% → 98.5% step validity, 2.1% → 92.0% trace validity), but you don’t foreground it enough.
  * Calibration analysis is mentioned but not really digested.
  * The zero-shot FOLIO results are treated as an afterthought instead of evidence that the option policy is not just overfitting to PrOntoQA.

* **Conclusion / Limitations**

  * Limitations are there but too minimal and generic.
  * You undersell the fact that you have an honest-to-God, closed-loop instantiation of OaK in a nontrivial domain.

Below is a restructured and rewritten version that fixes these issues while keeping your technical content and numbers.

---

### 2. Rewritten main text (LaTeX)

```latex
\begin{abstract}
Large language models (LLMs) often reach the right answer for the wrong reasons:
their chain-of-thought (CoT) rationales contain invalid inference steps, hidden
contradictions, or proof fragments that do not actually support the final
conclusion. Existing neuro-symbolic approaches either add soft logical
constraints to token-level predictions or invoke a symbolic solver to
post hoc verify free-form proofs, but they still treat reasoning as
unstructured text and do not learn which reasoning actions are reliable
in which contexts.

We introduce \sokrates{} (\textbf{S}ymbolic \textbf{O}ption-\textbf{K}nowledge
\textbf{R}easoning \textbf{A}lignment via \textbf{T}race \textbf{E}valuation
with \textbf{S}olver), a method that instantiates Sutton's Options and
Knowledge (\oak{}) framework in a first-order logic micro-world.
\sokrates{} represents proofs as sequences of discrete reasoning options:
macro-level inference rules such as \texttt{MODUS\_PONENS} or
\texttt{UNIV\_INST} with typed arguments, rather than free-form tokens.
A first-order logic solver provides ground-truth \emph{knowledge} by
checking each option application. From solver feedback we (i) train an
explicit option-success predictor $\qhat(s,\omega)$ and (ii) construct
preference pairs over optionized traces, applying Direct Preference
Optimization (\dpo{}) to align the LLM's option policy with
solver-induced preferences.

On PrOntoQA, \sokrates{} improves over supervised fine-tuning on
optionized traces, raising final accuracy from $94.2\%$ to $97.6\%$,
step validity from $27.3\%$ to $98.5\%$, and full-trace validity from
$2.1\%$ to $92.0\%$. The learned $\qhat$ is well calibrated under Brier
score and ECE, and the resulting option policy transfers zero-shot from
PrOntoQA to FOLIO, improving accuracy from $45.3\%$ to $53.2\%$.
\sokrates{} thus provides a concrete, data-efficient \oak{} loop for
symbolic reasoning: generate optionized traces, verify with a solver,
update predictive knowledge and policy, and repeat.
\end{abstract}

%============================================================================
\section{Introduction}
%============================================================================

Large language models have shown impressive multi-step reasoning
capabilities when prompted with chain-of-thought (CoT) explanations
\cite{wei2022chain} and sampled with self-consistency decoding
\cite{wang2023selfconsistency}. However, closer inspection reveals that
their intermediate reasoning is frequently wrong even when the final
answer is correct: proofs contain logically invalid steps, missing
premises, or contradictions that a symbolic checker would reject
\cite{saparov2023language,huang2024large}. This ``right answer, wrong
reasoning'' phenomenon makes such systems difficult to trust in
domains where intermediate steps must be sound.

Two families of neuro-symbolic methods attempt to close this gap.
Semantic loss approaches such as LoCo-LMs and Logical Neural Networks
\cite{riegel2020logical} incorporate differentiable logical constraints
into the training objective, encouraging local consistency of
token-level predictions but leaving the global proof process implicit.
Solver-augmented CoT approaches such as Logic-LM and LINC
\cite{pan2023logiclm,olausson2023linc} parse CoTs into first-order
logic (FOL) and invoke external theorem provers to verify or repair
them. These methods improve faithfulness, but they still treat
reasoning as unstructured text and do not learn explicit predictive
models of which reasoning \emph{actions} will succeed in which states.

We argue that logical reasoning is more naturally viewed as a
sequential decision problem over \emph{inference rules}: at each step,
the agent must decide which rule to apply and to which premises.
This perspective aligns with Sutton's Options and Knowledge (\oak{})
framework \cite{sutton2023reward}, which advocates that agents should
learn (1) a reusable vocabulary of temporally extended behaviors
(\emph{options}) and (2) explicit predictive models of how those
options behave (\emph{knowledge}). In a logic micro-world, inference
rules such as Modus Ponens or Universal Instantiation are precisely
such options, and a FOL solver can act as a source of ground-truth
knowledge about their success.

We introduce \sokrates{}, a system that instantiates this program for
LLM-based logical reasoning. Proofs are represented as sequences of
discrete options drawn from a small inference-rule vocabulary
(Table~\ref{tab:options}). A FOL solver checks each option application
and returns either a validated derived formula or an error. From this
signal we learn two coupled components:
(i) an option-success predictor $\qhat(s,\omega)$ that estimates the
probability that option $\omega$ will be solver-valid in state $s$;
and (ii) an option policy updated via solver-guided \dpo{}, which
builds preferences between better and worse traces based on
step-level validity and final correctness.
Figure~\ref{fig:architecture} summarizes this micro \oak{} loop.

Concretely, we make the following contributions:
\begin{itemize}
    \item \textbf{Optionized reasoning.} We cast proof construction as
    control over a finite option set of inference-rule macros with
    structured arguments, and generate reasoning traces in a
    Thought/Action format inspired by ReAct \cite{yao2023react}.
    \item \textbf{Explicit option knowledge.} We train an option-success
    head $\qhat(s,\omega)$ on solver labels, providing a calibrated,
    per-step estimate of logical validity for each option in context.
    \item \textbf{Solver-guided preference optimization.} We construct
    solver-derived preferences over optionized traces and apply
    \dpo{} \cite{rafailov2023direct} to align the option policy with
    stepwise logical correctness, yielding a concrete \oak{} loop for
    symbolic reasoning with LLMs.
\end{itemize}

On PrOntoQA, \sokrates{} substantially improves both task performance
and reasoning soundness: relative to an optionized supervised
fine-tuning (SFT) baseline, accuracy rises from $94.2\%$ to $97.6\%$,
while step validity jumps from $27.3\%$ to $98.5\%$ and the fraction of
fully valid traces from $2.1\%$ to $92.0\%$
(Table~\ref{tab:main_results}). The learned option policies also
transfer zero-shot to FOLIO, and the option head $\qhat$ is well
calibrated under Brier score and ECE, indicating that the model has
acquired nontrivial predictive knowledge about its own reasoning steps.

%============================================================================
\section{Background and Related Work}
%============================================================================

\subsection{LLM Reasoning and Failure Modes}

Chain-of-thought prompting \cite{wei2022chain} and self-consistency
decoding \cite{wang2023selfconsistency} are now standard tools for
eliciting multi-step reasoning from LLMs, but they offer no guarantees
that the resulting chains are logically valid. Systematic analyses
show that LLMs tend to be \emph{greedy reasoners}: they are often
locally competent at individual deductions but struggle to plan
globally coherent proofs when many valid next steps exist
\cite{saparov2023language}. Moreover, self-reflection without external
feedback frequently fails to repair such errors
\cite{huang2024large}.

\sokrates{} targets exactly this ``right answer, wrong reasoning''
regime by explicitly modeling which optionized reasoning actions are
solver-valid in which states, instead of relying on the surface
plausibility of free-form CoT text.

\subsection{Logical Reasoning Benchmarks}

\paragraph{Synthetic benchmarks.}
RuleTaker and ProofWriter \cite{clark2021transformers} provide
rule-based synthetic datasets with multi-hop proofs in controlled
worlds. PrOntoQA \cite{saparov2023language} extends this idea to
first-order logic with formally analyzable CoT reasoning, making it an
ideal testbed for studying step-level validity and trace faithfulness.

\paragraph{Natural language benchmarks.}
FOLIO \cite{han2022folio} consists of natural language premises
annotated with expert FOL translations and labels. P-FOLIO augments
FOLIO with human-written proof chains and rule labels, which inform
our choice of option vocabulary but are not directly used for
training.

We focus on PrOntoQA as our primary experimental domain because it
provides ground-truth proofs and fully specified FOL world models,
allowing us to parse and verify every optionized step with a solver.
We then evaluate zero-shot transfer to FOLIO.

\subsection{Neuro-Symbolic Reasoning and Solver-Augmented LMs}

\paragraph{LM + external solver at inference.}
LINC \cite{olausson2023linc} uses an LLM as a semantic parser that
produces FOL formulas, delegating proof search entirely to a symbolic
prover. Logic-LM \cite{pan2023logiclm} parses CoTs into FOL, verifies
them with a solver, and uses error messages to iteratively refine
reasoning. LAMBADA \cite{kazemi2022lambada} employs a backward
chaining control scheme, with LMs filling in missing premises or
subgoals. These approaches improve robustness by ``outsourcing'' logic
to a solver, but they do not learn an internal model of which
reasoning actions will succeed.

\paragraph{LMs trained to emulate solvers.}
LoGiPT \cite{feng2024logipt} trains LMs on hidden intermediate steps
from deductive solvers, so that the model can emulate the solver and
answer without external calls at test time. In contrast, \sokrates{}
does not aim to replace the solver entirely; instead, it uses the
solver as a source of labeled experience for learning an explicit
option-success model and solver-guided preferences over traces.

\paragraph{Neuro-symbolic consistency objectives.}
LoCo-LMs and Logical Neural Networks \cite{riegel2020logical}
introduce semantic loss functions that encourage consistency with
logical constraints at the prediction level. These methods reason over
truth values or soft logic constraints, rather than over a structured
sequence of symbolic options whose per-option success is modeled
explicitly.

\subsection{Preference Learning and Process Supervision}

Direct Preference Optimization (\dpo{}) \cite{rafailov2023direct}
provides an efficient alternative to RLHF with PPO
\cite{ouyang2022training}. Given a pair of responses $(y_w, y_l)$
where $y_w$ is preferred over $y_l$, \dpo{} optimizes
\begin{equation}
\mathcal{L}_{\text{DPO}} =
-\mathbb{E}\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right)
\right],
\label{eq:dpo}
\end{equation}
where $\pi_{\text{ref}}$ is a reference policy and $\beta$ controls
the strength of the implicit KL regularization.

Verifier-based process supervision has recently been applied to CoT
reasoning. VeriCoT \cite{ling2023deductive} translates CoTs to FOL,
verifies each step with a solver, and uses verification-based
preferences to fine-tune the model. However, VeriCoT operates directly
on unstructured CoT text and relies on a parser to extract predicates.
\sokrates{} instead works with a finite, typed option set
(Table~\ref{tab:options}), trains an explicit knowledge head
$\qhat(s,\omega)$ in parallel with \dpo{}, and frames training as a
micro \oak{} loop in which option models are learned as predictive
knowledge.

\subsection{Options, OaK, and Hierarchical RL}

The classic options framework \cite{sutton1999between} defines options
as temporally extended actions with an initiation set, an intra-option
policy, and a termination condition, enabling temporal abstraction and
planning. The \oak{} framework \cite{sutton2023reward} further
emphasizes that agents should learn not only control policies but also
\emph{knowledge}: predictive models about the effects of options.
Importantly, \oak{} advocates \emph{reward-respecting subtasks} whose
optimal policies do not conflict with the main objective.

From an \oak{} perspective, logical inference rules are options; the
FOL solver defines predictive knowledge about option outcomes; and
\sokrates{}'s \dpo{} update corresponds to improving the option policy
using this knowledge signal. Our subtask of ``maintain logical
consistency while answering the query'' is reward-respecting: the main
reward is final answer correctness, and the subtask reward is
solver-validated step correctness; these do not compete.

%============================================================================
\section{Problem Setup: OaK in a Logic World}
%============================================================================

We cast logical reasoning as a sequential decision problem in a FOL
micro-world where an agent incrementally constructs proofs by applying
inference-rule options.

\subsection{States and Goals}

Each problem instance defines:
\begin{itemize}
    \item A set of premises $P = \{p_1, \ldots, p_n\}$ in natural
    language with corresponding FOL forms.
    \item A target conclusion $c$ whose truth value under the world
    model is \textsc{True}, \textsc{False}, or \textsc{Unknown}.
\end{itemize}
At step $t$, the reasoning state is
$s_t = (P, D_t, c)$, where $D_t$ is the set of formulas derived so far
from applying inference rules:
$D_0 = \emptyset$ and $D_{t+1} = D_t \cup \{d_t\}$ if step $t$
produces a new formula $d_t$.

The goal is to decide whether $c$ is true, false, or undetermined in
the given world. In our experiments, this is a three-way classification
problem (true / false / unknown) with ground-truth labels provided by
the synthetic world model.

\subsection{Options as Inference Rules}

We define a finite option vocabulary $\Omega$ of inference-rule macros
(Table~\ref{tab:options}). Each option
$\omega \in \Omega$ consists of a rule type (e.g., Modus Ponens,
Universal Instantiation) and a small set of typed arguments such as
premise indices or constants. For example,
$\texttt{MODUS\_PONENS}(i,j)$ applies Modus Ponens to premise or
derived formula $i$ (containing $P$) and $j$ (containing $P \rightarrow Q$)
to infer $Q$.

Although each option is issued in a single decision step, it performs
multiple sub-operations: choosing the rule type, selecting indices of
premises or derived formulas, generating a natural language rationale,
and updating the proof state. In the \oak{} sense, options therefore
act as temporally extended cognitive macros.

\subsection{Knowledge: Solver as Ground Truth}

Given a state $s$ and an option $\omega$, a FOL solver (Z3 in our
experiments) checks whether applying $\omega$ yields a logically valid
new formula $d'$ that is entailed by the premises and derives:
\begin{equation}
    \textsc{Solver}(s, \omega) = \begin{cases}
        (\textsc{Valid}, d') & \text{if } \omega \text{ is logically valid} \\
        (\textsc{Invalid}, \emptyset) & \text{otherwise.}
    \end{cases}
    \label{eq:solver}
\end{equation}
This provides ground-truth \emph{knowledge} for (1) training the
option-success head $\qhat$ and (2) constructing \dpo{} preferences
over traces.

\subsection{Thought/Action Format}

Following ReAct \cite{yao2023react}, we format each reasoning step as
a \textbf{Thought/Action} pair. The \emph{Thought} describes, in
natural language, which premises and intermediate results are being
combined. The \emph{Action} is a structured XML-like option:
\begin{align*}
\texttt{Thought:} \;&\text{Stella is a wumpus (premise 2). Every wumpus is a tumpus (premise 0). So Stella is a tumpus.}\\
\texttt{Action:} \;&\texttt{<Option type="UNIV\_INST" args="[0, Stella]" />}
\end{align*}
Figure~\ref{fig:trace_example} shows a complete example trace with
solver annotations. This format separates free-form language used to
justify a step from the symbolic structure used for verification.

%============================================================================
\section{Method: \sokrates{}}
%============================================================================

\sokrates{} consists of three components that interact in an iterative
micro \oak{} loop (Algorithm~1):
(1) a generator that produces optionized Thought/Action traces from a
policy $\pi_\theta$;
(2) a solver that verifies each option application and provides labels
for $\qhat$; and
(3) a solver-guided \dpo{} update that improves the policy using
preference pairs constructed from verified traces.

\subsection{Optionized Trace Generation}

Given a problem $(s_0, c)$, we sample a trace from the current policy
$\pi_\theta$:

\begin{enumerate}
    \item Construct a prompt that lists numbered premises and the
    conclusion to be evaluated, and specifies the Thought/Action format
    and available rules.
    \item For $t = 1, \ldots, T_{\max}$:\footnote{We use
    $T_{\max} = 6$ in our experiments due to computational constraints;
    the full design targets $T_{\max} = 15$.}
    \begin{enumerate}
        \item Generate a \texttt{Thought} via unconstrained decoding.
        \item Generate an \texttt{Action} via grammar-constrained
        decoding, ensuring syntactic validity of the option and its
        arguments.
        \item Parse the action into an option $\omega_t$ and update
        the state to $s_{t+1}$.
        \item Terminate if $\omega_t = \texttt{CONCLUDE}$ or no valid
        next options remain.
    \end{enumerate}
    \item Return the trace
    $\tau = (s_0, \omega_1, s_1, \ldots, \omega_T, s_T)$.
\end{enumerate}

Constrained decoding ensures that syntactic errors (invalid option
formats) are eliminated by construction; semantic errors (logically
invalid rule applications) are detected by the solver.

\subsection{Solver Verification}

For each generated trace $\tau$, we apply the solver to every step and
record a binary validity label:
\begin{equation}
    v_t = \mathbf{1}[\textsc{Solver}(s_{t-1}, \omega_t) = \textsc{Valid}].
    \label{eq:step_valid}
\end{equation}
A trace is \textbf{fully valid} if all steps are solver-validated and
the final answer matches the ground-truth label:
\begin{equation}
    V(\tau) =
    \mathbf{1}\left[
    \left(\textstyle\prod_{t=1}^{T} v_t = 1\right)
    \land (\text{answer}(\tau) = \text{label})
    \right].
    \label{eq:trace_valid}
\end{equation}

\subsection{Option Success Predictor ($\qhat$)}

We attach an option-success head $\qhat(s,\omega)$ to the LLM that
predicts whether option $\omega$ will be solver-valid in state $s$:
\begin{equation}
    \qhat(s, \omega) =
    \sigma\left(
        \text{MLP}\left([\mathbf{h}_s; \mathbf{e}_\omega]\right)
    \right),
    \label{eq:qhat}
\end{equation}
where $\mathbf{h}_s$ is the hidden representation of the state
(e.g., the final token embedding in the prompt) and $\mathbf{e}_\omega$
is a learned embedding of the option type and arguments.

We train $\qhat$ with a binary cross-entropy loss on solver labels:
\begin{equation}
    \mathcal{L}_{\qhat} =
    -\mathbb{E}_{(s,\omega,v)}\left[
        v \log \qhat + (1{-}v) \log(1{-}\qhat)
    \right],
    \label{eq:qhat_loss}
\end{equation}
and evaluate its knowledge quality using Brier score and Expected
Calibration Error (ECE). A well-calibrated $\qhat$ allows the system
to quantify uncertainty over intermediate reasoning steps, not just
final answers.

\subsection{Preference Pair Construction}

From the verified traces, we construct preference pairs for \dpo{}.
For each problem, we generate $K$ traces and score each trace as
\begin{equation}
    \text{score}(\tau) =
    \frac{|\{t : v_t = 1\}|}{T}
    + \mathbf{1}[\text{correct}]
    + 0.5 \cdot \mathbf{1}[V(\tau) = 1],
    \label{eq:score}
\end{equation}
where the first term is the fraction of solver-valid steps, the
second term rewards correct final answers, and the third term gives a
bonus to fully valid traces.

We then select:
\begin{itemize}
    \item the \textbf{winner} $\tau_w$: the highest-scoring trace
    (ideally correct answer and all steps valid), and
    \item a \textbf{loser} $\tau_l$: a lower-scoring trace with either
    wrong answer or invalid steps.
\end{itemize}
Problems where all traces receive identical scores provide no
preference signal and are skipped. In practice, these account for
about $15\%$ of problems in the first iteration and about $5\%$ by the
second iteration as the policy improves.

\subsection{Micro OaK Training Loop}

Algorithm~1 summarizes the micro \oak{} loop. Starting from an SFT
model $\pi_0$ trained on optionized traces, we iterate:
\begin{enumerate}
    \item \textbf{Generate} $K$ traces per problem from current policy
    $\pi_{i-1}$.
    \item \textbf{Verify} each step with the solver to obtain validity
    labels $v_t$.
    \item \textbf{Update} the option-success head $\qhat$ on the
    resulting (state, option, label) triples using
    Eq.~\ref{eq:qhat_loss}.
    \item \textbf{Build preferences} $(\tau_w, \tau_l)$ using
    Eq.~\ref{eq:score}.
    \item \textbf{Apply \dpo{}} to update the policy from $\pi_{i-1}$
    to $\pi_i$ using the collected preference pairs.
\end{enumerate}
We use $N=2$ iterations in our main experiments due to computational
constraints. Conceptually, this constitutes a ``baby \oak{}'' loop:
experience (traces) $\rightarrow$ knowledge (solver labels, $\qhat$)
$\rightarrow$ policy improvement (\dpo{}) $\rightarrow$ richer
experience.

%============================================================================
\section{Experimental Setup}
%============================================================================

\subsection{Datasets}

\paragraph{PrOntoQA.}
We use the LoGiPT variant of PrOntoQA \cite{feng2024logipt}, which
contains $14{,}346$ training and $1{,}594$ test problems with proof
depths $1$--$5$ and varying numbers of distractor facts. Each problem
comes with a fully specified FOL world model and a ground-truth proof.

\paragraph{FOLIO.}
For transfer evaluation, we use FOLIO \cite{han2022folio}, which
contains $1{,}001$ training and $203$ validation examples with natural
language premises, expert FOL annotations, and labels.

\paragraph{Two-phase data strategy.}
We adopt a two-phase data usage scheme:
\begin{itemize}
    \item \textbf{SFT phase.} We use the full PrOntoQA training set
    ($n=14{,}346$) to train the initial optionized SFT model $\pi_0$,
    maximizing coverage of reasoning patterns.
    \item \textbf{\sokrates{} loop.} We run the \sokrates{} loop on a
    representative subset of $1{,}500$ problems ($\approx 10\%$ of the
    training set) to obtain solver-verified traces and preference
    pairs. This reflects that preference labels are more expensive
    than supervised traces.
\end{itemize}
Prior work on \dpo{} \cite{rafailov2023direct} suggests that preference
learning is sample-efficient, which our results corroborate.

\subsection{Models and Training}

Our base model is Qwen3-8B \cite{yang2024qwen2}, fine-tuned with
LoRA \cite{hu2022lora} (rank $r=64$, $\alpha=128$).

\paragraph{SFT.}
We perform supervised fine-tuning on optionized Thought/Action traces
for $3$ epochs with batch size $4$ (effective batch size $32$) and
learning rate $2 \times 10^{-5}$. This phase teaches the model the
option vocabulary and Thought/Action format but does not enforce
semantic correctness.

\paragraph{\dpo{} in the \sokrates{} loop.}
For each \oak{} iteration we train for $1$ epoch over the preference
pairs with \dpo{}, using $\beta = 0.1$ and learning rate
$5 \times 10^{-6}$. We use $N=2$ iterations and, for efficiency,
$K=2$ traces per problem with greedy decoding.

\paragraph{Generation hyperparameters.}
Table~\ref{tab:hyperparams} summarizes a hyperparameter sweep over
temperature $\tau$, maximum steps $T_{\max}$, and samples per problem
$K$. We find that moderate temperature $\tau = 0.5$ strikes a good
balance between accuracy and diversity for constructing preference
pairs: higher temperatures increase diversity but noticeably degrade
accuracy, while greedy decoding produces near-identical traces and
weak preferences.

\paragraph{Hardware.}
We train on $6\times$ NVIDIA B200 (183GB). SFT takes roughly ten
minutes; each \oak{} iteration takes approximately $45$–$60$ minutes
with distributed trace generation.

\subsection{Baselines}

We compare against:
\begin{itemize}
    \item \textbf{Base CoT.} Few-shot CoT prompting of Qwen3-8B on
    PrOntoQA, without fine-tuning.
    \item \textbf{Self-consistency.} Base CoT with $k=8$ independent
    samples and majority vote.
    \item \textbf{Optionized SFT.} Supervised fine-tuning on
    optionized Thought/Action traces without \sokrates{} preference
    learning.
\end{itemize}

\subsection{Metrics}

We evaluate at three levels:
\begin{itemize}
    \item \textbf{Task-level.} Final answer accuracy.
    \item \textbf{Proof-level.} Step validity (fraction of
    solver-valid steps) and trace validity (fraction of fully valid
    traces with all steps solver-validated and correct final answer).
    \item \textbf{Knowledge-level.} Brier score and ECE of the
    option-success head $\qhat$ relative to solver labels.
\end{itemize}

%============================================================================
\section{Results and Analysis}
%============================================================================

\subsection{Main Results on PrOntoQA}

Table~\ref{tab:main_results} reports results on the PrOntoQA test set
($n=1{,}594$). Without any fine-tuning, Qwen3-8B achieves $44.4\%$
accuracy with base CoT, improving to $53.8\%$ with self-consistency
voting ($k=8$). This confirms that raw CoT behavior is far from
solved on this benchmark.

Optionized SFT dramatically boosts accuracy to $94.2\%$, showing that
the model can learn to produce syntactically valid Thought/Action
traces when trained on ground-truth proofs. However, step validity
remains low ($27.3\%$) and trace validity is essentially zero
($2.1\%$): the model often reaches the correct final answer via
sequences of invalid reasoning steps.

Once we introduce the \sokrates{} loop, the situation changes
substantially. After a single \oak{} iteration, step validity jumps
to $87.8\%$ and trace validity to $71.3\%$, a $33\times$ increase in
fully valid traces relative to SFT. After a second iteration,
\sokrates{} reaches $97.6\%$ accuracy with $98.5\%$ step validity and
$92.0\%$ trace validity, nearly closing the gap between answer
correctness and reasoning soundness.

\subsection{Ablation Studies}

Table~\ref{tab:ablations} analyzes what components drive these gains.

\paragraph{Solver verification is essential.}
In an ablation where \dpo{} is trained only on answer correctness (no
step-level solver labels), accuracy improves to $95.5\%$, but step and
trace validity barely move ($31.6\%$ and $2.2\%$, respectively),
essentially matching the SFT baseline. This shows that the solver's
step-level feedback is crucial: without it, the model learns shortcuts
to the right answer while keeping structurally unsound proofs.

\paragraph{Number of \oak{} iterations.}
Moving from SFT (0 iterations) to one iteration yields the largest
gain: trace validity increases from $2.1\%$ to $71.3\%$. A second
iteration further improves trace validity to $92.0\%$. A third
iteration yields marginal additional gains in accuracy ($98.3\%$) and
slight changes in trace validity ($91.8\%$), indicating diminishing
returns.

\subsection{Calibration of the Option Head}

We assess whether $\qhat$ provides reliable knowledge by measuring
Brier score and ECE across models. The SFT-only model exhibits poor
calibration: predicted probabilities for step validity are
overconfident and poorly aligned with actual solver outcomes. After
running the \sokrates{} loop, both Brier score and ECE improve
substantially, and reliability diagrams (not shown here) indicate that
$\qhat$'s predicted probabilities closely track empirical success
rates across bins. This suggests that \sokrates{} learns not just to
produce valid steps more often, but also to estimate how likely a
candidate step is to be valid---a capability that could be exploited
for planning or search in future work.

\subsection{Zero-shot Transfer to FOLIO}

We evaluate zero-shot transfer by training models only on PrOntoQA
and evaluating them directly on FOLIO without any additional
fine-tuning (Table~\ref{tab:transfer}). FOLIO is substantially harder
than PrOntoQA: premises are natural language rather than templated,
and proofs are longer and structurally more varied.

Base CoT and self-consistency both achieve $42.9\%$ accuracy on
FOLIO. Transferring the optionized SFT model yields a modest
improvement to $45.3\%$ accuracy, with $46.5\%$ step validity and
$9.9\%$ trace validity. In contrast, the \sokrates{} model trained on
PrOntoQA reaches $53.2\%$ accuracy with $48.3\%$ step validity and
$14.8\%$ trace validity. These gains indicate that the option policy
and knowledge learned in the synthetic FOL micro-world provide
nontrivial benefits when applied to richer natural language reasoning
tasks, even without domain-specific tuning.

%============================================================================
\section{Conclusion and Limitations}
%============================================================================

We presented \sokrates{}, a neuro-symbolic method that instantiates
the \oak{} framework for LLM-based logical reasoning. By representing
proofs as sequences of inference-rule options, using a FOL solver as a
source of predictive knowledge about option success, and applying
solver-guided \dpo{} in an iterative micro \oak{} loop, \sokrates{}
substantially improves both accuracy and reasoning soundness on
PrOntoQA, while learning a calibrated option-success model and
transferring zero-shot to FOLIO.

\paragraph{Limitations and future work.}
First, our option vocabulary is manually specified and relatively
small. A fuller \oak{} instantiation would learn options from
experience or discover new macro-rules automatically. Second, we
apply \qhat{} only during training as an auxiliary head; we do not yet
use it for planning or test-time search (e.g., to prune low-probability
options or guide Tree-of-Thoughts-style exploration). Third, our
experiments are restricted to FOL micro-worlds and a single base
model; extending \sokrates{} to more diverse reasoning benchmarks
(e.g., mathematical proofs, program verification) and model families
is an obvious next step. Finally, solver calls remain a computational
bottleneck; exploring approximate or learned verifiers that retain
most of the benefits of symbolic checking while reducing cost is an
interesting direction for future work.

\bibliography{references}

%============================================================================
\appendix
\section{Prompt and Generation Details}
\label{app:prompt}
%============================================================================

% (Keep your existing appendix and figures here.)
```

This gives you a structurally coherent, narrative-driven paper with:

* A sharper problem statement and positioning.
* A clear mapping to OaK (options + knowledge).
* Stronger and more explicit contrasts to prior work.
* A focused experimental story that emphasizes step/trace validity, calibration, and transfer.

You can now plug your existing tables, figures, and appendix back in; only the prose and equations above replace your current draft.
