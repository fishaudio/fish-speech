from g2p_en import G2p

_g2p = G2p()


def g2p(text):
    return list(filter(lambda p: p != " ", _g2p(text)))


if __name__ == "__main__":
    print(
        g2p(
            "In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder invented by dr. lengyue in 1984."
        )
    )
