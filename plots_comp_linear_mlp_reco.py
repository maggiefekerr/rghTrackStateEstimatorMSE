import sys
import datetime
import uproot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Comparing p, theta, phi extracted from linear fit reconstruction (method of Silvia) to 
# p, theta, phi extracted from multi-layer perceptron reconstruction (method of Tongtong)

root_file = "/w/hallb-scshelf2102/clas12/mkerr/rgh/krishna_ana/output/ana_recoil_45deg_45deg_mctrue_recoil_989.root"
csv_file  = "/w/hallb-scshelf2102/clas12/mkerr/rgh/reco_ai/maggie_model/output/45deg_45deg/truth_vs_pred_45deg_45deg_v1.csv"
end_label = "45deg_45deg_v1"

# retrieving information from csv
def unpack_csv():
    d = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    p_tr = d[:,0] 
    theta_tr = d[:,1] 
    phi_tr = d[:,2]
    p_pred = d[:,3]
    theta_pred = d[:,4] 
    phi_pred = d[:,5]
    return p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred
#enddef

# retrieving tree
def get_tree():
    return (uproot.open(root_file)["Recoil"])
#enddef

# retrieving mask
def get_mask(tr):
    p_lin     = tr["p_tof"].array(library="np")
    p_tr      = tr["p_prot"].array(library="np")
    theta_lin = tr["theta_prot_det"].array(library="np")
    theta_tr  = tr["theta_prot"].array(library="np")
    phi_lin   = tr["phi_prot_det"].array(library="np")
    phi_tr    = tr["phi_prot"].array(library="np")
    mask      = (p_tr > 0.) & (np.isfinite(theta_tr)) & (np.isfinite(phi_tr))
    mask      = mask & (p_lin > 0.) & (np.isfinite(theta_lin)) & (np.isfinite(phi_lin))
    return mask
#enddef

# gaussian function
def gaussian(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
#enddef

# plotting linear reconstruction versus truth
def _plot_lin_reco(tr, mask):
    # branches
    p_lin        = tr["p_tof"].array(library="np")
    p_lin_tr     = tr["p_prot"].array(library="np")
    theta_lin    = tr["theta_prot_det"].array(library="np")
    theta_lin_tr = tr["theta_prot"].array(library="np")
    phi_lin      = tr["phi_prot_det"].array(library="np")
    phi_lin_tr   = tr["phi_prot"].array(library="np")

    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_lin[mask], bins=80, range=(0,3.0), density=True, histtype="stepfilled", color="darkgreen", alpha=0.45, label="Linear Reco")
    axes[0].hist(p_lin_tr[mask], bins=80, range=(0,3.0), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[0].set_xlabel(r"p (GeV)")
    axes[0].set_ylabel(r"Density")
    axes[0].set_title(r"Momentum")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_lin[mask], bins=80, range=(0,100), density=True, histtype="stepfilled", color="darkgreen", alpha=0.45, label="Linear Reco")
    axes[1].hist(theta_lin_tr[mask], bins=80, range=(0,100), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[1].set_xlabel(r"$\theta$ ($^\circ$)")
    axes[1].set_ylabel(r"Density")
    axes[1].set_title(r"Lab Theta")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_lin[mask], bins=80, range=(-180,180), density=True, histtype="stepfilled", color="darkgreen", alpha=0.45, label="Linear Reco")
    axes[2].hist(phi_lin_tr[mask], bins=80, range=(-180,180), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[2].set_xlabel(r"$\phi$ ($^\circ$)")
    axes[2].set_ylabel(r"Density")
    axes[2].set_title(r"Lab Phi")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/linear_reco_vs_truth_"+end_label+".pdf")
    plt.close()
#enddef

# plotting multi-layer perceptron reconstruction versus truth
def _plot_mlp_reco(p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred):
    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_pred, bins=80, range=(0,3.0), density=True, histtype="stepfilled", color="indigo", alpha=0.45, label="MLP Reco")
    axes[0].hist(p_tr, bins=80, range=(0,3.0), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[0].set_xlabel(r"p (GeV)")
    axes[0].set_ylabel(r"Density")
    axes[0].set_title(r"Momentum")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_pred, bins=80, range=(0,100), density=True, histtype="stepfilled", color="indigo", alpha=0.45, label="MLP Reco")
    axes[1].hist(theta_tr, bins=80, range=(0,100), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[1].set_xlabel(r"$\theta$ ($^\circ$)")
    axes[1].set_ylabel(r"Density")
    axes[1].set_title(r"Lab Theta")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_pred, bins=80, range=(-180,180), density=True, histtype="stepfilled", color="indigo", alpha=0.45, label="MLP Reco")
    axes[2].hist(phi_tr, bins=80, range=(-180,180), density=True, histtype="step", color="black", linewidth=2, label="MC Truth")
    axes[2].set_xlabel(r"$\phi$ ($^\circ$)")
    axes[2].set_ylabel(r"Density")
    axes[2].set_title(r"Lab Phi")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/mlp_reco_vs_truth_"+end_label+".pdf")
    plt.close()
#enddef

# plotting linear reco-truth
def _plot_lin_diff(tr, mask):
    # branches
    p_lin        = tr["p_tof"].array(library="np")
    p_lin_tr     = tr["p_prot"].array(library="np")
    p_diff       = p_lin-p_lin_tr
    theta_lin    = tr["theta_prot_det"].array(library="np")
    theta_lin_tr = tr["theta_prot"].array(library="np")
    theta_diff   = theta_lin-theta_lin_tr
    phi_lin      = tr["phi_prot_det"].array(library="np")
    phi_lin_tr   = tr["phi_prot"].array(library="np")
    phi_diff     = phi_lin-phi_lin_tr

    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_diff[mask], bins=80, range=(-1,1), histtype="stepfilled", color="darkgreen", alpha=0.45, label=r"$\mu$="+f"{np.mean(p_diff[mask]):.4f}"+r" GeV"+"\n"+
                                                                                                                      r"$\sigma$="+f"{np.std(p_diff[mask]):.4f}"+r" GeV")
    axes[0].axvline(x=0.0, linestyle="--", color="black")
    axes[0].set_xlabel(r"p$_{\text{reco}}$ - p$_{\text{truth}}$ (GeV)")
    axes[0].set_ylabel(r"Counts")
    axes[0].set_title(r"Momentum Residual")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_diff[mask], bins=80, range=(-45,45), histtype="stepfilled", color="darkgreen", alpha=0.45, label=r"$\mu$="+f"{np.mean(theta_diff[mask]):.4f}"+r"$^\circ$"+"\n"+
                                                                                                                        r"$\sigma$="+f"{np.std(theta_diff[mask]):.4f}"+r"$^\circ$")
    axes[1].axvline(x=0.0, linestyle="--", color="black")
    axes[1].set_xlabel(r"$\theta_{\text{reco}}$ - $\theta_{\text{truth}}$ ($^\circ$)")
    axes[1].set_ylabel(r"Counts")
    axes[1].set_title(r"Lab Theta Residual")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_diff[mask], bins=80, range=(-25,25), histtype="stepfilled", color="darkgreen", alpha=0.45, label=r"$\mu$="+f"{np.mean(phi_diff[mask]):.4f}"+r"$^\circ$"+"\n"+
                                                                                                                      r"$\sigma$="+f"{np.std(phi_diff[mask]):.4f}"+r"$^\circ$")
    axes[2].axvline(x=0.0, linestyle="--", color="black")
    axes[2].set_xlabel(r"$\phi_{\text{reco}}$ - $\phi_{\text{truth}}$ ($^\circ$)")
    axes[2].set_ylabel(r"Counts")
    axes[2].set_title(r"Lab Phi Residual")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/linear_diff_"+end_label+".pdf")
    plt.close()
#enddef

# plotting mlp reco-truth
def _plot_mlp_diff(p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred):
    p_diff       = p_pred-p_tr
    theta_diff   = theta_pred-theta_tr
    phi_diff     = phi_pred-phi_tr

    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_diff, bins=80, range=(-1,1), histtype="stepfilled", color="indigo", alpha=0.45, label=r"$\mu$="+f"{np.mean(p_diff):.4f}"+r" GeV"+"\n"+
                                                                                                         r"$\sigma$="+f"{np.std(p_diff):.4f}"+r" GeV")
    axes[0].axvline(x=0.0, linestyle="--", color="black")
    axes[0].set_xlabel(r"p$_{\text{reco}}$ - p$_{\text{truth}}$ (GeV)")
    axes[0].set_ylabel(r"Counts")
    axes[0].set_title(r"Momentum Residual")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_diff, bins=80, range=(-45,45), histtype="stepfilled", color="indigo", alpha=0.45, label=r"$\mu$="+f"{np.mean(theta_diff):.4f}"+r"$^\circ$"+"\n"+
                                                                                                               r"$\sigma$="+f"{np.std(theta_diff):.4f}"+r"$^\circ$")
    axes[1].axvline(x=0.0, linestyle="--", color="black")
    axes[1].set_xlabel(r"$\theta_{\text{reco}}$ - $\theta_{\text{truth}}$ ($^\circ$)")
    axes[1].set_ylabel(r"Counts")
    axes[1].set_title(r"Lab Theta Residual")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_diff, bins=80, range=(-25,25), histtype="stepfilled", color="indigo", alpha=0.45, label=r"$\mu$="+f"{np.mean(phi_diff):.4f}"+r"$^\circ$"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(phi_diff):.4f}"+r"$^\circ$")
    axes[2].axvline(x=0.0, linestyle="--", color="black")
    axes[2].set_xlabel(r"$\phi_{\text{reco}}$ - $\phi_{\text{truth}}$ ($^\circ$)")
    axes[2].set_ylabel(r"Counts")
    axes[2].set_title(r"Lab Phi Residual")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/mlp_diff_"+end_label+".pdf")
    plt.close()
#enddef

# plotting linear and mlp reco
def _plot_comp_lin_mlp_reco(tr, mask, p_mlp, theta_mlp, phi_mlp):
    # branches for linear
    p_lin          = tr["p_tof"].array(library="np")
    theta_lin      = tr["theta_prot_det"].array(library="np")
    phi_lin        = tr["phi_prot_det"].array(library="np")

    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_lin[mask], bins=80, range=(0.0, 3.0), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco")
    axes[0].hist(p_mlp, bins=80, range=(0.0, 3.0), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco")
    axes[0].set_xlabel(r"p (GeV)")
    axes[0].set_ylabel(r"Density")
    axes[0].set_title(r"Momentum")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_lin[mask], bins=80, range=(0.0, 100.0), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco")
    axes[1].hist(theta_mlp, bins=80, range=(0.0, 100.0), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco")
    axes[1].set_xlabel(r"$\theta$ ($^\circ$)")
    axes[1].set_ylabel(r"Density")
    axes[1].set_title(r"Lab Theta")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_lin[mask], bins=80, range=(-180,180), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco")
    axes[2].hist(phi_mlp, bins=80, range=(-180,180), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco")
    axes[2].set_xlabel(r"$\phi$ ($^\circ$)")
    axes[2].set_ylabel(r"Density")
    axes[2].set_title(r"Lab Phi")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/lin_mlp_reco_"+end_label+".pdf")
    plt.close()
#enddef

# plotting linear and mlp reco-truth
def _plot_comp_lin_mlp_diff(tr, mask, p_tr_mlp, theta_tr_mlp, phi_tr_mlp, p_mlp, theta_mlp, phi_mlp):
    # branches for linear
    p_lin          = tr["p_tof"].array(library="np")
    p_tr_lin       = tr["p_prot"].array(library="np")
    p_diff_lin     = p_lin-p_tr_lin
    theta_lin      = tr["theta_prot_det"].array(library="np")
    theta_tr_lin   = tr["theta_prot"].array(library="np")
    theta_diff_lin = theta_lin-theta_tr_lin
    phi_lin        = tr["phi_prot_det"].array(library="np")
    phi_tr_lin     = tr["phi_prot"].array(library="np")
    phi_diff_lin   = phi_lin-phi_tr_lin

    # for mlp
    p_diff_mlp       = p_mlp-p_tr_mlp
    theta_diff_mlp   = theta_mlp-theta_tr_mlp
    phi_diff_mlp     = phi_mlp-phi_tr_mlp

    # plotting
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.0,5.0))
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.12, top=0.9)
    # p
    axes[0].hist(p_diff_lin[mask], bins=80, range=(-1,1), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco\n"+r"$\mu$="+f"{np.mean(p_diff_lin[mask]):.4f}"+r" GeV"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(p_diff_lin[mask]):.4f}"+r" GeV")
    axes[0].hist(p_diff_mlp, bins=80, range=(-1,1), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco\n"+r"$\mu$="+f"{np.mean(p_diff_mlp):.4f}"+r" GeV"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(p_diff_mlp):.4f}"+r" GeV")
    axes[0].axvline(x=0.0, linestyle="--", color="black")
    axes[0].set_xlabel(r"p$_{\text{reco}}$ - p$_{\text{truth}}$ (GeV)")
    axes[0].set_ylabel(r"Density")
    axes[0].set_title(r"Momentum Residual")
    axes[0].legend(loc="upper right")
    # theta
    axes[1].hist(theta_diff_lin[mask], bins=80, range=(-45,45), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco\n"+r"$\mu$="+f"{np.mean(theta_diff_lin[mask]):.4f}"+r"$^\circ$"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(theta_diff_lin[mask]):.4f}"+r"$^\circ$")
    axes[1].hist(theta_diff_mlp, bins=80, range=(-45,45), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco\n"+r"$\mu$="+f"{np.mean(theta_diff_mlp):.4f}"+r"$^\circ$"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(theta_diff_mlp):.4f}"+r"$^\circ$")
    axes[1].axvline(x=0.0, linestyle="--", color="black")
    axes[1].set_xlabel(r"$\theta_{\text{reco}}$ - $\theta_{\text{truth}}$ ($^\circ$)")
    axes[1].set_ylabel(r"Density")
    axes[1].set_title(r"Lab Theta Residual")
    axes[1].legend(loc="upper right")
    # phi
    axes[2].hist(phi_diff_lin[mask], bins=80, range=(-25,25), density=True, histtype="step", linewidth=2, color="darkgreen", label="Linear Reco\n"+r"$\mu$="+f"{np.mean(phi_diff_lin[mask]):.4f}"+r"$^\circ$"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(phi_diff_lin[mask]):.4f}"+r"$^\circ$")
    axes[2].hist(phi_diff_mlp, bins=80, range=(-25,25), density=True, histtype="step", linewidth=2, color="indigo", label="MLP Reco\n"+r"$\mu$="+f"{np.mean(phi_diff_mlp):.4f}"+r"$^\circ$"+"\n"+
                                                                                                             r"$\sigma$="+f"{np.std(phi_diff_mlp):.4f}"+r"$^\circ$")
    axes[2].axvline(x=0.0, linestyle="--", color="black")
    axes[2].set_xlabel(r"$\phi_{\text{reco}}$ - $\phi_{\text{truth}}$ ($^\circ$)")
    axes[2].set_ylabel(r"Density")
    axes[2].set_title(r"Lab Phi Residual")
    axes[2].legend(loc="upper right")
    plt.savefig("/work/clas12/mkerr/rgh/reco_ai/maggie_model/plots/lin_mlp_comp_"+end_label+".pdf")
    plt.close()
#enddef

# main method
def main():
    # info from multi-layer perceptron reconstruction
    p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred = unpack_csv()

    # info from linear reconstruction
    tr = get_tree()
    mask = get_mask(tr)
    p_lin        = tr["p_tof"].array(library="np")
    p_lin_tr     = tr["p_prot"].array(library="np")
    theta_lin    = tr["theta_prot_det"].array(library="np")
    theta_lin_tr = tr["theta_prot"].array(library="np")
    phi_lin      = tr["phi_prot_det"].array(library="np")
    phi_lin_tr   = tr["phi_prot"].array(library="np")

    # plots for linear and mlp reconstruction
    _plot_lin_reco(tr, mask)
    _plot_mlp_reco(p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred)

    # plots for linear and mlp difference between reconstruction and truth
    _plot_lin_diff(tr, mask)
    _plot_mlp_diff(p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred)

    # compare linear and mlp difference between reconstruction and truth
    _plot_comp_lin_mlp_diff(tr, mask, p_tr, theta_tr, phi_tr, p_pred, theta_pred, phi_pred)
    _plot_comp_lin_mlp_reco(tr, mask, p_pred, theta_pred, phi_pred)
#enddef

main()