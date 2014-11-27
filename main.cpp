#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include <cstdio>
#include <cmath>
#include <cstring>

#define MAX_FEATURES 25
#define NUM_EPOCHS 15

// constants

double LRATE_ub = 0.012;
double LRATE_mb = 0.003;

double LAMBDA_ub = 0.03;
double LAMBDA_mb = 0.001;

double LEARNING_RATE_USER = 0.006;
double LEARNING_RATE_MOVIE = 0.011;

const double LEARNING_RATE_DECREASE_RATE = 0.9;

double K_user = 0.08;
double K_movie = 0.006;

double b_u[10001]; //базовых предикторы отдельных пользователей
double b_m[10001]; //базовых предикторы отдельных фильмов
double mu = 0; //общий средний рейтинг по базе

//Переменный для оценки среднеквадратической ошибки
double RMSE = 1;
double OLD_RMSE = 0;
double threshold = 0.01;

double err = 0; //Функция ошибки
double eta = 0.1; //Скорость обучения

double u_f[MAX_FEATURES][10001]; //вектор факторов пользователя
double m_f[MAX_FEATURES][10001]; //вектор факторов фильмов

int k; //максимальная оценка
int U; //число пользователей
int M; //число фильмов
int D; //кол. элементов обучающей выборки
int T; //кол. элементов тестовой выборки

double dot(int u, int m);
void result();

struct Container {
    int MovieID;
    int CustomerID;
    double Rating;
} Real[1000001];


int main()
{

    //READ
    scanf("%d%d%d%d%d", &k, &U, &M, &D, &T);
    memset(Real, 0, sizeof Real);
    int u, m, r;
    for(int i = 0; i < D; ++i){
        scanf("%d%d%d", &u, &m, &r);
        Real[i].CustomerID = u;
        Real[i].MovieID = m;
        Real[i].Rating = (double)r;
    }
    //---------------

    //Initialize
    memset(b_u, 0, sizeof b_u);
    memset(b_m, 0, sizeof b_m);

    for(int i = 0; i < MAX_FEATURES; ++i)
        for (int j =0; j < U; ++j)
            u_f[i][j] = 0.1;

    for(int i = 0; i < MAX_FEATURES; ++i)
        for(int j = 0; j < M; ++j)
            m_f[i][j] = 0.1;

    //---------------

    //Learn
    int iterations = 0;
    while(iterations < NUM_EPOCHS || abs(OLD_RMSE - RMSE) > 0.00001)
    {
        OLD_RMSE = RMSE;
        RMSE = 0;
        for(int i = 0; i < D; ++i){
            double p = mu + b_u[Real[i].CustomerID] + b_m[Real[i].MovieID] + dot(Real[i].CustomerID, Real[i].MovieID);
            if (p > k) p = k;
            if (p < 2) p = 2;

            err = Real[i].Rating - p;
            RMSE = err * err;

            mu += eta*err;
            b_u[Real[i].CustomerID] += LRATE_ub*(err - LAMBDA_ub*b_u[Real[i].CustomerID]);
            b_m[Real[i].MovieID] += LRATE_mb*(err - LAMBDA_mb*b_m[Real[i].MovieID]);

            for(int f = 0; f < MAX_FEATURES; ++f){
                u_f[f][Real[i].CustomerID] += LEARNING_RATE_USER*(err*m_f[f][Real[i].MovieID] - K_user*u_f[f][Real[i].CustomerID]);
                m_f[f][Real[i].MovieID] += LEARNING_RATE_MOVIE*(err*u_f[f][Real[i].CustomerID] - K_movie*m_f[f][Real[i].MovieID]);
            }
        }
        RMSE = sqrt(RMSE/D);
        // decrease the learning rates

        LEARNING_RATE_USER *= LEARNING_RATE_DECREASE_RATE;
        LEARNING_RATE_MOVIE *= LEARNING_RATE_DECREASE_RATE;
        LRATE_ub *= LEARNING_RATE_DECREASE_RATE;
        LRATE_mb *= LEARNING_RATE_DECREASE_RATE;

        if (RMSE > OLD_RMSE - threshold){
            eta = eta * 0.66;
            threshold = threshold * 0.5;
        }
        ++iterations;
    }
    //---------------
    result();
    return 0;
}

double dot(int u, int m)
{
    double res = 0;
    for(int i = 0; i < MAX_FEATURES; ++i)
        res += u_f[i][u]*m_f[i][m];
    return res;
}

void result()
{
    int u, m;
    double r;
    for(int i = 0; i < T; ++i)
    {
        scanf("%d%d", &u, &m);
        r = mu + b_u[u] + b_m[m] + dot(u, m);
        if (r > k) r = k;
        if (r < 1) r = 1;
        printf("%.6lf\n", r);
    }
}
