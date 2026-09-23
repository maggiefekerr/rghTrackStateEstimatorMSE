import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))


class Plotter:
    def __init__(self, print_dir="", end_name=""):
        self.print_dir = print_dir
        self.end_name = end_name
        self.state_parameter_name = ["p", "theta", "phi", "particleType"]
        os.makedirs(self.print_dir, exist_ok=True)

    # ------------------------------------------------------------
    # training loss
    # ------------------------------------------------------------
    def plotTrainLoss(self, tracker):

        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        print("Final train loss:", train_losses[-1])
        print("Final val loss:", val_losses[-1])

        plt.figure(figsize=(10, 6))

        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        outname = os.path.join(self.print_dir, f"loss_{self.end_name}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

    # ------------------------------------------------------------
    # residual
    # ------------------------------------------------------------
    def plot_residuals(self, preds, targets, fit_range_factor={"p": 0.8, "theta": 0.8, "phi": 0.4, "particleType": 0.8}):

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        plot_ranges = {
            'p': (-0.25, 0.25),
            'theta': (-5, 5),
            'phi': (-20, 20),
            'particleType': (-100, 100)
        }

        # --------------------------------------------------------
        # Fit range factors
        # --------------------------------------------------------
        if isinstance(fit_range_factor, (int, float)):
            fit_range_factors = {
                "p": fit_range_factor,
                "theta": fit_range_factor,
                "phi": fit_range_factor,
                "particleType": fit_range_factor
            }
        else:
            fit_range_factors = {
                "p": fit_range_factor.get("p", 2.0),
                "theta": fit_range_factor.get("theta", 2.0),
                "phi": fit_range_factor.get("phi", 2.0),
                "particleType": fit_range_factor.get("particleType", 2.0)
            }

        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        axes = axes.flatten()

        fit_results = []

        for i, name in enumerate(self.state_parameter_name):

            # ---------------------------------------
            # Residuals
            # ---------------------------------------
            diff = preds[:, i] - targets[:, i]
            diff = diff[np.isfinite(diff)]

            plot_min, plot_max = plot_ranges[name]

            # Initial estimates from all residuals
            mu0 = np.mean(diff)
            sigma0 = np.std(diff)

            factor = fit_range_factors[name]

            fit_min = mu0 - factor * sigma0
            fit_max = mu0 + factor * sigma0

            # ---------------------------------------
            # Histogram
            # ---------------------------------------
            counts, bins, _ = axes[i].hist(
                diff,
                bins=60,
                range=(plot_min, plot_max),
                color="royalblue",
                alpha=0.7,
                label="Residuals"
            )

            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Fit only within mean ± 2σ
            mask = (
                    (bin_centers >= fit_min)
                    & (bin_centers <= fit_max)
                    & (counts > 0)
            )

            x_fit = bin_centers[mask]
            y_fit = counts[mask]

            if len(x_fit) < 5:
                fit_results.append((name, np.nan, np.nan))
                axes[i].set_title(f"{name} residual")
                continue

            # Initial guess
            A0 = np.max(y_fit)

            # Poisson uncertainties
            sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

            try:
                popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, mu0, sigma0], sigma=sigma_y, absolute_sigma=True,
                                       bounds=([0.0, fit_min, 1e-8], [np.inf, fit_max, np.inf]))

                A, mu, sigma = popt

                fit_results.append((name, mu, sigma))

                # Draw fitted Gaussian
                x_curve = np.linspace(plot_min, plot_max, 500)
                y_curve = gaussian(x_curve, A, mu, sigma)

                axes[i].plot(x_curve, y_curve, "r--", lw=3,
                             label=(f"Gaussian Fit:\n" f"μ={mu:+.3e}\n" f"σ={sigma:.3e}")
                             )

                # Show fit window
                axes[i].axvline(fit_min, color="green", ls=":", lw=1.5)
                axes[i].axvline(fit_max, color="green", ls=":", lw=1.5)

            except RuntimeError:

                fit_results.append((name, np.nan, np.nan))

                axes[i].text(0.05, 0.95, "Fit failed", transform=axes[i].transAxes, color="red", fontsize=11,
                             verticalalignment="top")

            axes[i].set_title(f"{name} residual")
            axes[i].set_xlabel("Prediction − Truth")
            axes[i].set_ylabel("Counts")
            axes[i].set_xlim(plot_min, plot_max)
            axes[i].legend(loc="upper left")

        # Hide unused subplot
        for j in range(len(self.state_parameter_name), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        outname = os.path.join(
            self.print_dir,
            f"track_diff_{self.end_name}.png"
        )
        plt.savefig(outname, dpi=200)
        plt.close()

        # --------------------------------------------------
        # Save Gaussian fit results to CSV
        df = pd.DataFrame(fit_results, columns=['Variable', 'Mean (μ)', 'Sigma (σ)'])
        csv_path = os.path.join(self.print_dir, f"track_diff_fit_{self.end_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved fit results to {csv_path}")

    # ------------------------------------------------------------
    # prediction vs target
    # ------------------------------------------------------------
    def plot_pred_target(self, preds, targets, bins=100):
        """
        2D prediction vs target plots.

        Parameters
        ----------
        preds : torch.Tensor
            shape (N,4)

        targets : torch.Tensor
            shape (N,4)

        bins : int
        """

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        plt.figure(figsize=(12, 4))

        for dim in range(4):
            plt.subplot(1, 4, dim + 1)

            xmin = min(targets[:, dim].min(), preds[:, dim].min())
            xmax = max(targets[:, dim].max(), preds[:, dim].max())

            plt.hist2d(targets[:, dim], preds[:, dim], range=[[xmin, xmax], [xmin, xmax]], bins=bins, norm=colors.LogNorm())

            plt.plot([xmin, xmax], [xmin, xmax], "r--",linewidth=2)

            plt.xlabel("Target")
            plt.ylabel("Prediction")
            plt.title(self.state_parameter_name[dim])

            plt.colorbar()

        plt.tight_layout()

        outname = os.path.join( self.print_dir, f"pred_vs_target_{self.end_name}.png")

        plt.savefig(outname, dpi=200, bbox_inches="tight")

        plt.close()
