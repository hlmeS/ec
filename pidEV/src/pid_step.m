function resp = pid_step(Kp, Ki, Kd, u, t)

s = tf('s');
G = 1/(s^2 + 10*s + 20);

%Kp = 300;
%Ki = 350;
%Kd = 50;
C = pid(Kp,Ki,Kd);
T = feedback(C*G,1);

[y, t] = lsim(T, u, t);
resp = [y, t];

%t = 0:0.05:15;
%stepinfo(sys);
%[y,t] = step(sys);
%resp = [t,y];
%Y = y';
%T = t';
%resp = Y;

