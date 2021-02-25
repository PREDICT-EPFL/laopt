#include <iostream>
#include <string>
#include <utility>

template <class ClassT, class... ArgsT>
auto getCallbackPtr(ClassT* obj, void(ClassT::* memfn)(ArgsT...))
{
    return [obj, memfn](ArgsT&&... args) {
        (obj->*memfn)(std::forward<ArgsT>(args)...);
    };
}
template <typename memFn, class ClassT>
auto getCallbackTemplate(ClassT* obj)
{
    return [obj](auto&&... args){
        return (obj->memFn)(std::forward<decltype(args)>(args)...);
    };
}
template <typename memFn, class ClassT, class... ArgsT>
auto getCallbackRedundant(ClassT* obj)
{
    return [obj](ArgsT&&... args){
        return (obj->memFn)(std::forward<ArgsT&&>(args)...);
    };
}

// Example of use
class Foo {
public:
    void bar(size_t& x, const std::string& s) { x=s.size(); }
};
int main() {
    Foo f; 
    auto c1 = getCallbackPtr(&f, &Foo::bar);
    size_t x1; c1(x1, "123"); std::cout << "c1:" << x1 << "\n";
    auto c2 = getCallbackTemplate<&Foo::bar>(&f);
    size_t x2; c2(x2, "123"); std::cout << "c2:" << x2 << "\n";
    auto c3 = getCallbackRedundant<&Foo::bar, Foo, size_t&, const std::string&>(&f);
    size_t x3; c3(x3, "123"); std::cout << "c3:" << x3 << "\n";
}