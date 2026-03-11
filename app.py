abstract class Shape{

    abstract double calculateArea();

    void displayShape(){
        System.out.println("Shape Area Calculation");
    }
}

class Circle extends Shape{

    double radius = 5;

    double calculateArea(){
        return Math.PI * radius * radius;
    }
}

class Rectangle extends Shape{

    double length = 4;
    double width = 6;

    double calculateArea(){
        return length * width;
    }
}

public class Main{

    public static void main(String[] args){

        Circle c = new Circle();
        c.displayShape();
        System.out.println("Circle Area: " + c.calculateArea());

        System.out.println();

        Rectangle r = new Rectangle();
        r.displayShape();
        System.out.println("Rectangle Area: " + r.calculateArea());
    }
}
