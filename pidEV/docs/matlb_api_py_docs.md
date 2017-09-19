# MATLAB API for python

## collection of API calls MATLAB and python

``` python
import matlab.engine

eng = matlab.engine.start_matlab()

import StringIO
out = StringIO.StringIO()
err = StringIO.StringIO()

eng.pid_script(300, 350, 50, nargout = 0)
eng.workspace['Y']

output = eng.pid_step(350, 300, 50), stdout=out, stderr=err)
np.asarray(output)
```


``` matlab
% Matlab :
t = 0:0.01:5;
sys = tf(1, [1 10 20]);
opt = stepDataOptions('InputOffset', 5, 'StepAmplitude', 5);
step(sys, t, opt)

```
