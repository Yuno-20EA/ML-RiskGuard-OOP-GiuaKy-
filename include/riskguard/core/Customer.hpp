#pragma once

namespace riskguard {

class Customer {
public:
    Customer(double inc, double db, double del, int a)
        : income(inc), debt(db), delinquency(del), age(a) {}

    double get_income() const { return income; }
    void set_income(double v) { income = v; }

    double get_debt() const { return debt; }
    void set_debt(double v) { debt = v; }

    double get_delinquency() const { return delinquency; }
    void set_delinquency(double v) { delinquency = v; }

    int get_age() const { return age; }
    void set_age(int v) { age = v; }

private:
    double income{0.0};
    double debt{0.0};
    double delinquency{0.0};
    int age{0};
};

} // namespace riskguard
