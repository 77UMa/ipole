#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#define SLOW_LIGHT (0)

// Model-specific definitions and globals
#define KRHO 0
#define UU   1
#define U1   2
#define U2   3
#define U3   4
#define B1   5
#define B2   6
#define B3   7
#define KEL  8
// --- [修改开始] ---
#define UNTH 9    // [新增] 对应非热电子归一化常数 C
#define P_IDX 10  // [新增] 对应谱指数 p
#define KTOT 11   // [修改] 将变量总数从 9 改为 11
#define GAMMA_MIN_IDX 12  // [新增] 最小洛伦兹因子 gamma_min (Bug 2 修复)
// --- [修改结束] ---
#define TFLK 8  // temperature of fluid in Kelvin
#define THF  8  // fluid temperature in me c^2

extern double DTd;
extern double sigma_cut;

#endif // MODEL_PARAMS_H
