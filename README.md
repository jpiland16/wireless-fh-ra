# wireless-fh-ra
python implementation of [*Game Theoretic Anti-jamming Dynamic Frequency
Hopping and Rate Adaptation in Wireless Systems*](https://uweb.engr.arizona.edu/~mjabdelrahman/WiOpt14.pdf) by Hanawal, Abdel-Rahman, and Krunz

## Open questions
 - The two changes [here](https://github.com/jpiland16/wireless-fh-ra/commit/b33c13bae73708b75a8e10fab709cf92f78cab57#diff-4a0ba52b538019d585d8daad98157504254f8da1bbbfda6df1aef2e21fcf61db) don't seem to reflect anything in the original paper, but were necessary in order to avoid errors where the jammer went too long without guessing the correct channel. This could be an issue with my code.
 - The first two changes [here](https://github.com/jpiland16/wireless-fh-ra/commit/e83725cfa774d05906500f2b310f99a6fe237590#diff-fada037ad086638e65c7ae77e3d223963e9afaa26326aab0ea718f4013176e43), (1) avoiding where `x` might be `'j'` and (2) avoiding where `self.params.n * x` might be less than or equal to `self.params.k`, also seemed to be a bit of a hack to get the optimization to work.
 - When running with `TIME_AHEAD = 0`, my results seem similar to the first graph presented in the paper. However, this is not true reinforcement learning. increasing the `TIME_AHEAD` parameter seems to decrease the performance of the transmitter, which signals some sort of error in my implementation. (more research needed)
 - My understanding of the paper was that when the transmitter sends at the lowest rate, the jammer cannot interrupt the transmission. Is this the case?