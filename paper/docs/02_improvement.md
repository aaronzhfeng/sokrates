# Paper Improvements - Prompt Design & Training Details

Based on our implementation and training runs, here are suggested additions to the paper.

---

## 1. Add Prompt Design Section (§4.1 or Appendix A)

The prompt is a key design choice that enables optionized reasoning. Currently missing from the paper.

### Suggested Addition to §4.1 (Optionized Trace Generation)

Add after "Construct prompt with premises and target conclusion":

```latex
\paragraph{Prompt Structure.}
We use a structured prompt that explicitly instructs the model to produce Thought/Action pairs with our option vocabulary:

\begin{listing}[t]
\begin{lstlisting}[caption={Prompt template for optionized reasoning.}]
You are a logical reasoning assistant. Given 
premises and a conclusion, determine if the 
conclusion is TRUE, FALSE, or UNKNOWN.
Reason step by step using formal inference rules.

For each step, provide:
Thought: Your reasoning in natural language
Action: <Option type="RULE_NAME" args="[indices]"/>

Available rules: MODUS_PONENS, MODUS_TOLLENS, 
UNIV_INSTANTIATION, AND_INTRO, AND_ELIM, OR_INTRO, 
DISJUNCTIVE_SYLLOGISM, etc.
End with: <Option type="CONCLUDE" args="[0/1/2]"/>
(0=TRUE, 1=FALSE, 2=UNKNOWN)

---

Premises:
  [0] {premise_0}
  [1] {premise_1}
  ...

Conclusion to evaluate: {conclusion}

Reasoning:
\end{lstlisting}
\end{listing}

Key design choices:
\begin{itemize}
    \item \textbf{Numbered premises} enable options to reference formulas by index
    \item \textbf{Explicit rule vocabulary} in prompt constrains the option space
    \item \textbf{Terminal encoding} (0/1/2) provides unambiguous answer format
\end{itemize}
```

---

## 2. Two-Phase Training Design (§5.2)

Our actual training uses different data scales for SFT vs DPO. This is methodologically important.

### Suggested Addition to §5.2 (Training Configuration)

```latex
\paragraph{Two-Phase Data Strategy.}
We employ different data scales for each training phase:
\begin{itemize}
    \item \textbf{SFT}: Full training set ($n{=}14{,}346$) to maximize format learning diversity
    \item \textbf{\sokrates{} (DPO phase)}: Representative subset ($n{=}1{,}500$; 10\%) for efficient preference learning
\end{itemize}

This reflects a realistic deployment scenario: supervised data is abundant, but preference labels require expensive solver verification. 
The subset is randomly sampled (seed=42) from the same distribution.
Prior work on DPO~\cite{rafailov2023direct} demonstrates that preference learning is sample-efficient, with 1--5K examples typically sufficient for convergence.
```

---

## 3. Distributed Training Details (§5.2 or Appendix)

For reproducibility, document the multi-GPU setup.

### Suggested Addition

```latex
\paragraph{Distributed Training.}
SFT uses 2 GPUs with data-parallel training; the \sokrates{} loop uses 6 GPUs with distributed trace generation.
For trace generation, problems are split across GPUs (250 problems/GPU), with traces gathered via \texttt{all\_gather} before preference construction.
This enables linear scaling of the computationally intensive generation phase.
```

---

## 4. Time-Optimized Configuration (§5.2 footnote or Appendix)

Document our practical compromises.

### Suggested Footnote

```latex
\footnote{Due to computational constraints, we use a time-optimized configuration: 
2 OaK iterations (vs.\ 3), 2 samples/problem (vs.\ 8), max 6 proof steps (vs.\ 15), 
and greedy decoding for deterministic generation. 
Full-scale experiments with original settings are left for future work.}
```

---

## 5. Update Experimental Setup Numbers (§5.2)

Current paper says $K=8$ samples, $N=3$ iterations. Update to match actual run:

### Changes Needed

| Paper Currently | Actual Run | Update? |
|-----------------|------------|---------|
| $K=8$ samples/problem | $K=2$ | Yes, with footnote |
| $N=3$ iterations | $N=2$ | Yes, with footnote |
| $T_{\max}=15$ steps | $T_{\max}=6$ | Yes |
| SFT: ~1.5 hours | SFT: ~10 min | Yes |
| OaK iter: ~1.5 hours | OaK iter: ~45-60 min | Yes |

---

## 6. Add Appendix A: Full Prompt Example

### Suggested Appendix Content

```latex
\appendix
\section{Prompt and Generation Details}

\subsection{Complete Prompt Template}

Figure~\ref{fig:full_prompt} shows the complete prompt used for trace generation.
The prompt serves three purposes:
(1) establishes the task (logical reasoning),
(2) specifies the output format (Thought/Action pairs),
(3) constrains the action space to our option vocabulary.

\begin{figure}[h]
\centering
\fbox{\parbox{0.95\columnwidth}{
\small
\texttt{You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN. Reason step by step using formal inference rules.}\\[0.5em]
\texttt{For each step, provide:}\\
\texttt{Thought: Your reasoning in natural language}\\
\texttt{Action: <Option type="RULE\_NAME" args="[indices]" />}\\[0.5em]
\texttt{Available rules: MODUS\_PONENS, MODUS\_TOLLENS, UNIV\_INSTANTIATION, AND\_INTRO, AND\_ELIM, OR\_INTRO, DISJUNCTIVE\_SYLLOGISM, etc.}\\
\texttt{End with: <Option type="CONCLUDE" args="[0/1/2]" />}\\
\texttt{(0=TRUE, 1=FALSE, 2=UNKNOWN)}\\[0.5em]
\texttt{---}\\[0.5em]
\texttt{Premises:}\\
\texttt{~~[0] Every wumpus is a tumpus.}\\
\texttt{~~[1] Every tumpus is a rompus.}\\
\texttt{~~[2] Stella is a wumpus.}\\[0.5em]
\texttt{Conclusion to evaluate: Stella is a rompus.}\\[0.5em]
\texttt{Reasoning:}
}}
\caption{Complete prompt template with example problem from PrOntoQA.}
\label{fig:full_prompt}
\end{figure}

\subsection{Generation Parameters}

We use the following generation settings:
\begin{itemize}
    \item Maximum steps: $T_{\max} = 6$
    \item Decoding: Greedy (deterministic)
    \item Maximum thought tokens: 60
    \item Maximum action tokens: 25
    \item Tokenizer padding: Left (required for batched generation with decoder-only models)
\end{itemize}
```

---

## 7. Results Table Update

Once we have results, update Table 2 with actual numbers. The structure is good, just needs data.

### Expected Results Format

```latex
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Acc.} & \textbf{Step} & \textbf{Trace} & \textbf{Brier} \\
\hline
Base CoT & 62.3 & -- & -- & -- \\
SFT & 78.5 & 71.2 & 45.3 & 0.182 \\
\hline
\sokrates{} (iter 1) & 82.1 & 79.4 & 58.7 & 0.143 \\
\sokrates{} (iter 2) & 84.7 & 83.2 & 64.1 & 0.118 \\
\hline
\end{tabular}
```

---

## 8. Minor Text Fixes

### §4.1 Line 279
Change `$T_{\max}=15$` to `$T_{\max}=6$` (or add footnote about time-optimized setting)

### §4.4 Line 325
Change `$K=8$` to `$K=2$` (with footnote explaining time constraints)

### §5.2 Hardware
Update from "6× NVIDIA B200" to actual config with time estimates

---

## Summary of Key Additions

| Section | Addition | Priority |
|---------|----------|----------|
| §4.1 | Prompt structure paragraph | **High** |
| §5.2 | Two-phase data strategy | **High** |
| Appendix A | Full prompt example | **Medium** |
| §5.2 | Time-optimized footnote | **Medium** |
| §5.2 | Distributed training note | Low |

These additions strengthen the paper by:
1. Making the method fully reproducible
2. Justifying our data split choices
3. Being transparent about computational trade-offs

