from control import *

pxs = 400
summary = False

control = RL(
    IMG_W = pxs,
    IMG_H = pxs,
    latent_dim = 500
)

control.createCAE(summary=False)
control.testAE()

control.autoencoder.predict()