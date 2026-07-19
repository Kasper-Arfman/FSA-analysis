# What a typical FOV looks like

# What a typical ROI looks like
# + bead
# - bead

# DNA detection
# - 3D gaussian blurring
# - Dynamic thresholding

# Length measurement
# > Bead:
# - track the bead explicity
# > Free
# - track only the DNA




# Oscillation of known flow rate ramp
# Use the bead + free pair as a flow/Force sensor


F_free = F_WLC(Lf) = f_DNA*v
F_bead = F_WLC(Lb) = f_DNA*v + f_bead*v

Two equations with two unknowns (f_DNA, v)
We can test the assumption f_DNA << f_bead

So in order to fairly reject f_DNA, the bead size must be ...













# > Validate that the flow sensor works