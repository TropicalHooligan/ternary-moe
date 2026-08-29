#ifndef ROUTER_H
#define ROUTER_H

int router_top1(const float *scores, int n);
void router_top2(const float *scores, int n, int *i0, int *i1);

#endif