#pragma once
#include <string>

namespace riskguard {

class Customer {
public:
    Customer() = default;
    
    Customer(int age, double income, double credit_score, std::string job_status)
        : age(age), income(income), credit_score(credit_score), job_status(std::move(job_status)) {}

    // Các phương thức Getters và Setters
    int get_age() const { return age; }
    void set_age(int a) { age = a; }

    double get_income() const { return income; }
    void set_income(double inc) { income = inc; }

    double get_credit_score() const { return credit_score; }
    void set_credit_score(double score) { credit_score = score; }

    const std::string& get_job_status() const { return job_status; }
    void set_job_status(const std::string& status) { job_status = status; }

private:
    int age{0};
    double income{0.0};
    double credit_score{0.0};
    std::string job_status;
};

} // namespace riskguard
