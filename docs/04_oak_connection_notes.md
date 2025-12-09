You’re right: taking OaK seriously *does* shift the design. The current OaK-DPO idea is “OaK-flavored,” but not really an OaK **architecture** yet. Let me lay out exactly how and what we should change.

---

## 1. What OaK actually insists on

From Sutton’s OaK / FC-STOMP work, there are a few non-negotiables:

1. **Options are learned behaviours**, not just a manually fixed library.
2. **Knowledge = predictive models** of options (option models / GVFs), used for planning.
3. **Reward-respecting subtasks:** subtasks are defined so that pursuing them doesn’t wreck the main reward, and each subtask is solved to produce an option.
4. **A continual loop:** features → subtasks → options → models → planning → new features, run online, not as a one-shot offline pre-train.

So OaK is not just “we have options and a model.” It’s a *process* that keeps generating new subtasks/options/models from experience.

---

## 2. Where our current OaK-DPO design deviates

Right now, OaK-DPO (as we framed it) is:

* **Options:** fixed, hand-defined macros = inference rules (`MODUS_PONENS`, `UNIV_INSTANTIATION`, …).
* **Knowledge:**

  * external: solver;
  * internal: LM policy over options after DPO (implicit “knowledge”).
* **Learning:** one or a few offline passes: sample traces → solver labels valid/invalid → build preference pairs → DPO.

So:

* Options are **not discovered** from reward-respecting subtasks; they’re imposed.
* Knowledge is **not explicit** option models; it’s implicit in the LM.
* There’s **no continual loop**; just offline distillation.

That’s fine for a workshop paper, but if you want to honestly say “this is an OaK instance,” we need to move a bit closer to the real thing.

---

## 3. How this should shift the design (concretely)

### A. Make it a micro-OaK loop in a logical world

Shift your story from “one-shot alignment” to “a small but genuine OaK cycle”:

1. **Environment:** FOLIO / P-FOLIO / PrOntoQA become your *world*. Each problem is an episode; the solver is ground truth about dynamics.
2. **Experience:** at each iteration, you roll out optionized proofs with the current LM, get solver feedback (valid/invalid steps), and log trajectories.
3. **Learning updates:** from that stream you:

   * update the option policy (via DPO), and
   * update explicit option models (see next point).

Do this in a few outer loops rather than one big static dataset. That’s your “baby OaK” continual cycle.

### B. Add explicit option models (real “knowledge”)

Right now, “knowledge” is only implicit. Make it explicit:

* Train a small head (q_\phi(s, \omega)) to predict **probability that option (\omega)** will be solver-valid in state (s) (using embeddings of state + option).
* Supervise (q_\phi) directly from solver labels over all sampled steps.
* Evaluate calibration (Brier / ECE) before vs after OaK-DPO.

Then in the paper you can honestly say:

> For each option we learn a predictive model of its success probability, i.e., a simple option model in the OaK sense.

You still use DPO to shape the policy, but you now *also* have explicit “knowledge” to point to.

### C. Introduce “reward-respecting” subtasks in the logic domain

Right now, your only “reward” is: final proof valid or not. To be closer to OaK:

* Define **auxiliary subtasks** like:

  * “reach any state where predicate X is derived,”
  * “complete a proof without ever deriving a contradiction,”
  * “close all open quantifiers.”
* Reward per subtask = small bonus when its condition is met, *plus* original task reward; no penalty for steps as long as you don’t break consistency. That’s the logic analog of **reward-respecting feature-attainment subtasks**.
* For each such subtask, you:

  * define an option (policy in proof space),
  * learn a small value head for that subtask (GVF).

You don’t need full RL here; you can learn these heads from logged trajectories in a supervised / off-policy way. But conceptually, you now *have* reward-respecting subtasks → options → models.

### D. Make DPO one component, not the whole story

DPO remains your main “policy improvement” mechanism (it’s practical and on-trend), but the design shifts from:

> “Solver → DPO → better CoT”

to:

> “Experience → solver labels →
> (i) option models & GVFs (knowledge),
> (ii) DPO policy update over options (behaviour),
> repeated in a small OaK loop.”

That’s much closer to STOMP/FC-STOMP: SubTask → Option → Model → Planning, embedded in a feedback loop.

---

## 4. What does *not* need to change (for feasibility)

You don’t need to:

* Implement full online RL with average reward and off-policy control.
* Implement feature discovery or meta-learning the way Sutton dreams about; we can admit that’s future work.
* Abandon DPO or your solver-guided preference idea.

For the workshop, a realistic plan is:

1. Keep the **optionized CoT + solver + DPO** core.
2. Add **explicit option-success heads** and measure them (that’s your “knowledge”).
3. Run at least **two or three outer cycles** of generate → verify → re-train, to show an embryonic continual loop.
4. Add a couple of **auxiliary, reward-respecting subtasks** (consistency, specific predicate attainment) with small GVF heads.

Then you can legitimately say: “This is not full OaK, but it is a concretely instantiated OaK-style STOMP cycle in a logical micro-environment. We demonstrate that symbolic knowledge, expressed via solver feedback, can be distilled into both option policies and simple option models in an LM.”

If you want, next step is: I rewrite the paper’s **“OaK connection”** section in this revised, more honest way so you can drop it straight into a draft.
