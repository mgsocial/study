
# class 정리

# 상속(inheritance) 이란?
# 클래스에서의 상속이란, 물려주는 클래스의 내용(속성과 메소드)을 물려받는 클래스가 가지게 되는 것

## 자식클래스에서는 부모클래스의 속성과 메소드는 기재하지 않아도 포함이 된다

class Country:
    """Super Class"""
    name = '국가명'
    population = '인구'
    capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')

class Korea(Country):
    """Sub Class"""
    def __init__(self, name):
        self.name = name

    def show_name(self):
        print('국가 이름은 : ', self.name)


a = Korea('대한민국')
a.show()
a.show_name()
print(a.capital)
print(a.name)

# 메소드 오버라이딩
# : 부모 클래스의 매소드를 자식 클래스에서 재정의 하는 것

class Korea(Country):

    def __init__(self, name, population, capital):
        self.name = name
        self.population = population
        self.capital = capital

    def show(self):
        print(
            """
            국가의 이름은 {} 입니다.
            국가의 인구는 {} 입니다.
            국가의 수도는 {} 입니다.
            """.format(self.name, self.population, self.capital)
        )

a = Korea('대한민국', 50000000, '서울')
a.show()

class Korea(Country):
    def __init__(self, name, population, capital):
        self.name = name
        self.population = population
        self.capital = capital

    def show(self):
        super().show()
        print(
            """
            국가의 이름은 {} 입니다.
            국가의 인구는 {} 입니다.
            국가의 수도는 {} 입니다.
            """.format(self.name, self.population, self.capital)
        )

a = Korea('대한민국', 50000000, '서울')
a.show()

# 다중상속

class Country:
    """Super Class"""
    name = '국가명'
    population = '인구'
    capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')

class Province:
    Province_list = []


class Korea(Country, Province):
    def __init__(self, name, population, capital):
        self.name = name
        self.population = population
        self.capital = capital

    def show(self):
        super().show()
        print(
            """
            국가의 이름은 {} 입니다.
            국가의 인구는 {} 입니다.
            국가의 수도는 {} 입니다.
            """.format(self.name, self.population, self.capital)
        )

a = Korea('대한민국', 50000000, '서울')
a.show()

print(Korea.mro())

