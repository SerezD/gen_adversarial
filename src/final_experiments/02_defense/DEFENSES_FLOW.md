## FLOW OF EXPERIMENTS!

The following experiments are written in an easy-to-hard order, meaning that any correctly formulated defense
should perform worse and worse. 

#### Random and General Attacks:

- clean accuracy
- Apply rotations and translations.
- Apply common corruptions and perturbations.
- Add Gaussian noise of increasingly large standard deviation.
- Attack with random noise of the correct norm (brute force of N different random noises).

#### Unbounded Attacks - Gradient-Free and Hard-Label:

- Examples are NES and Boundary Attacks.
- Try both targeted and un-targeted.
- Plot success rate vs perturbation rate until reaching 100 % success rate.

#### Gradient-Based Attacks:

- Few but different and effective attacks.
- Use EOT since we have a randomized defense. 
- Try both targeted and un-targeted.
- Reached sufficient iterations to converge (plot iterations vs success rate).
- Sufficient random restarts done to avoid suboptimal local minima (pick best result).
- Increasing the perturbation increases success rate (plot perturbation vs success, until random guessing).
- Explore different choices of the step size or other attack hyperparameters.

#### Compare to similar Previous Work:

- Check RobustML or other libraries, pick similar defenses.
- Attempt attacks that are **similar** (not the same!) to those that defeated previous similar defenses.
- When comparing against prior work, ensure it has not been broken.

#### Ablation Studies:

- try different components (remove randomness and take only mean.)
- re-sample different combinations of latents (these should perform worse)

#### Transfer Attacks:

- NVAE 3x1 to NVAE 3x4.
- Another model with same hyperparams but different seeds.

#### Adaptive Attack:

- What attack could possibly defeat this defense?
- One that is forced to found perturbations in the first latents!
- Can give an upper bound of robustness.