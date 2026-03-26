import numpy as np

class ChannelSelector:
    def __init__(self, n_channels=5, seed=0, y_init=np.zeros((2,1)), u_init=np.zeros((2,1))):
        self.n_channels = n_channels
        self.seed = seed
        np.random.seed(self.seed)
        self.probs = np.random.rand(self.n_channels)
        self.u = u_init
        self.y = y_init
        self.y_last = y_init
        self.u_last = u_init
        self.round = 0
        self.success = True

    def send_u(self, u, channel=0):
        self.u = u
        self.success = np.random.rand() <= self.probs[channel]
        return self.success

    def send_y(self, y):
        self.y = y

    def get_u(self):
        if self.success:
            self.u_last = self.u
            return self.u
        else:
            return self.u_last

    def get_y(self):
        if self.success:
            self.y_last = self.y
            return self.y
        else:
            return self.y_last


if __name__ == "__main__":
    y = np.random.randn(2, 1)
    u = np.random.randn(2, 1)
    channel_selector = ChannelSelector(n_channels=5, seed=0, y_init=y, u_init=u)

    for i in range(2500):
        #### Controller side #####
        y = channel_selector.get_y()
        u = np.random.randn(2, 1)
        # Select channel
        channel = np.random.choice(channel_selector.n_channels)
        success = channel_selector.send_u(u, channel=channel)


        ##### System side #####
        u = channel_selector.get_u()
        y = np.random.randn(2, 1)
        channel_selector.send_y(y)
