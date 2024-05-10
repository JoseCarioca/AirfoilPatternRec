function [si,no]=ainb(x,y)

x=tocol(x);
y=torow(y);

A=repmat(x,1,length(y));
B=repmat(y,length(x),1);

T=sum(A==B);

si=find(T~=0);
no=find(T==0);

