
#include <stdio.h>

#define N 20

int main() {
    double x[N] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double y[N] = {35,39,44,50,56,61,67,72,78,83,89,94,100,105,111,116,122,127,133,138};

    double w = 0.0;
    double b = 0.0;

    double learning_rate = 0.001;
    int epochs = 5000;

    double initial_loss = 0.0;

    for (int i = 0; i < N; i++) {
        double y_pred = w * x[i] + b;
        initial_loss += (y[i] - y_pred) * (y[i] - y_pred);
    }
    initial_loss /= N;

    printf("Initial Weight = %.4lf\n", w);
    printf("Initial Bias   = %.4lf\n", b);
    printf("Initial Loss   = %.4lf\n\n", initial_loss);

    for (int epoch = 0; epoch < epochs; epoch++) {

        double dw = 0.0;
        double db = 0.0;

        for (int i = 0; i < N; i++) {
            double y_pred = w * x[i] + b;
            double error = y_pred - y[i];

            dw += error * x[i];
            db += error;
        }

        dw = (2.0 / N) * dw;
        db = (2.0 / N) * db;

        w = w - learning_rate * dw;
        b = b - learning_rate * db;
    }

    double final_loss = 0.0;

    for (int i = 0; i < N; i++) {
        double y_pred = w * x[i] + b;
        final_loss += (y[i] - y_pred) * (y[i] - y_pred);
    }

    final_loss /= N;

    printf("Final Weight = %.4lf\n", w);
    printf("Final Bias   = %.4lf\n", b);
    printf("Final Loss   = %.4lf\n", final_loss);

    printf("\nFinal Learned Equation:\n");
    printf("y = %.4lf * x + %.4lf\n\n", w, b);

    printf("Predictions:\n");
    printf("-----------------------------\n");
    printf("x\tActual\tPredicted\n");

    for (int i = 0; i < N; i++) {
        double y_pred = w * x[i] + b;
        printf("%.0lf\t%.0lf\t%.2lf\n", x[i], y[i], y_pred);
    }

    return 0;
}
