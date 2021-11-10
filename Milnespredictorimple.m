clear all,close all,clc

%question partA b
h=0.1;
t1=1;
t=0:h:t1;
y2(1)=2;
N=t1/h;

for i=1:N
    grad=-10*y2(i)+10*(t(i)*t(i)) + 2*t(i);
    y2(i+1)=y2(i) +h*grad;
end

hold on
plot(t,y2,'-*')
    

%using ode45 to find the exact solution 
odefunc=@(t,y) (-10*y+10*t^2+2*t);
tspan=[0 t1];
y0=2;

[t_out,y_out]=ode45(odefunc,tspan,y0)
plot(t_out,y_out,'-+')

absolute_error=abs(y2-y_out);
absolute_error


%there is minor difference in the variation between the exact and the
%approximted values using the Nystrom-milne method,this is justified by the
%absolute error values which converges as the time t increases



%partA c

h=0.1;
t1=1;
t=0:h:t1;
y2(1)=2;
N=t1/h;

odefunc=@(t,y) (-10*y+10*t^2+2*t);
tspan=[0 t1];
y0=2;

[t_out,y_out]=ode113(odefunc,tspan,y0)
plot(t_out,y_out,'-*')

%the resulting solution of y  decreases its value upto a certain time t then it starts to 
%increases its values exponentilly as time t increases

%partA d
h=0.1;
t1=1;
t=0:h:t1;
y2(1)=2;
N=t1/h;

for i=1:N
    grad=-10*y2(i)+10*(t(i)*t(i)) + 2*t(i);
    y2(i+1)=y2(i) +h*grad;
end

hold on
plot(t,y2,'-*')
    

%using ode45 to find the exact solution 
odefunc=@(t,y) (-10*y+10*t^2+2*t);
tspan=[0 t1];
y0=2;

[t_out,y_out]=ode45(odefunc,tspan,y0)
plot(t_out,y_out,'-+')

legend('Nyström-Milne','ode113')
%as time t increases exponentially so is both the results of y function
%from the ode113 and Nystrom-milne ,which in time seems to converge at some
%high t value
