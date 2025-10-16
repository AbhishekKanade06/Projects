package Cus_Service;
import java.sql.*;
import java.time.LocalDate;
import java.util.Scanner;
import HOLTE.Hotel;

public class Services extends Hotel{
    Scanner sc=new Scanner(System.in);
    String Service_name;
    int services;
    int service_count=1;
    int price;
    LocalDate date=LocalDate.now();
    public void get_services(Connection con,int customerId) throws SQLException {
        display_Service(con);
        System.out.println("\n1.Food\n2.Wifi\n3.Gym\n4.Laundry\n5.Car_rent\n\t");
        System.out.println("\tEnter 0 for exit");
        System.out.print("select your Services : ");
        services=sc.nextInt();
        while(services!=0&&service_count<=5){
            switch(services){
                case 1:
                    System.out.print("Food Service : \n1.Food_1\n2.Food_3\n3.Food_3\n");
                    switch(sc.nextInt()){
                        case 1:
                            Service_name="Food_1";
                            break;
                        case 2:
                            Service_name="Food_2";
                            break;
                        case 3:
                            Service_name="Food_3";
                            break;
                        default:
                            break;            
                    }
                    set_service(con, Service_name, customerId);
                    System.out.println(Service_name+" is selected");
                    break;
                case 2:
                    Service_name="WiFi"; 
                    set_service(con, Service_name, customerId);
                    System.out.println(Service_name+" is added");    
                    break;
                case 3:
                    Service_name="Gym";
                    set_service(con, Service_name, customerId);
                    System.out.println(Service_name+" is added");
                    break;
                case 4:
                    Service_name="Laundry";
                    set_service(con, Service_name, customerId);
                    System.out.println(Service_name+" is added");
                    break;  
                case 5:
                    Service_name="Car";
                    set_service(con, Service_name, customerId);
                    System.out.println(Service_name+" is added");
                    break; 

            }
            System.out.println("\tEnter 0 for exit");
            System.out.print("select your Services : ");
            services=sc.nextInt();
            service_count++;

        }
    }
    public static void display_Service(Connection con) throws SQLException{
        String q5="select * from Service";
        PreparedStatement ps=con.prepareStatement(q5);
        ResultSet re=ps.executeQuery();
        System.out.printf("|%-6s|%-15s|%-6s|%n","Sr_no","Service","Price");
        System.out.printf("%-27s %n", "-------------------------------");
        while(re.next()){
             System.out.printf("|%-6s|%-15s|%-6s|%n",re.getInt("Sr_no"),re.getString("Service_name"),re.getInt("Price"));
        }
        System.out.println();
    }
    public void create_bill(Connection con,int customerId) throws SQLException{
        String q="create table bill_cus_"+customerId+"(Sr_no int primary key auto_increment,Srevice varchar(20) unique,Price int,Date_of_start date,Date_of_end date,Total_price int)";
        PreparedStatement ps=con.prepareStatement(q);
        ps.executeUpdate();
        
    }
    public void set_service(Connection con,String Service_name,int customerId) throws SQLException{
        String q="select * from Service where Service_name=?";
        PreparedStatement p=con.prepareStatement(q);
        p.setString(1, Service_name);
        ResultSet r=p.executeQuery();
        while(r.next()){
            price=r.getInt("Price");
        }
        String q1="insert into bill_cus_"+customerId+"(Srevice,Price,Date_of_start) values(?,?,?)";
        PreparedStatement p1=con.prepareStatement(q1);
        p1.setString(1, Service_name);
        p1.setInt(2, price);
        p1.setObject(3,date);
        p1.executeUpdate();
    }
    public void get_bill(Connection con,int customerId) throws SQLException{
        show_details(con, customerId);
        String q="update bill_cus_"+customerId+" set Total_price=((select datediff(Date_of_end,Date_of_start)+1)*Price)";
        PreparedStatement ps=con.prepareStatement(q);
        ps.executeUpdate();
        String q1="select * from bill_cus_"+customerId;
        PreparedStatement ps1=con.prepareStatement(q1);
        ResultSet rs1=ps1.executeQuery();
        System.out.printf("|%-6s|%-15s|%-6s|%n","Sr_no","Service","Price");
        System.out.printf("%-27s %n", "-------------------------------");
        while (rs1.next()) {
             System.out.printf("|%-6s|%-15s|%-6s|%n",rs1.getInt("Sr_no"),rs1.getString("Srevice"),rs1.getInt("Total_price"));
            System.out.printf("%-27s %n", "-------------------------------");
        }
        String q2="select sum(Total_price) as price from bill_cus_"+customerId;
        PreparedStatement ps2=con.prepareStatement(q2);
        ResultSet rs2=ps2.executeQuery();
        while (rs2.next()) {
            System.out.printf("%-27s %n", "-----------------------------");
            System.out.printf("|%-27s| %n", "Net Price is : "+rs2.getInt("price"));
            System.out.printf("%-27s %n", "-----------------------------");
        }
    }
    public  void show_details(Connection con,int customerId) throws SQLException{
        System.out.println("\nService Bill ");
        String q="select * from customer where customerId=?";
        PreparedStatement ps=con.prepareStatement(q);
        ps.setInt(1, customerId);
        ResultSet rs=ps.executeQuery();
        while(rs.next()){
            System.out.println("\nName : "+rs.getString("customerName"));
            System.out.println("ContactNo : "+rs.getString("contactNo"));
            System.out.println(rs.getString("IdName")+"No. :"+rs.getString("IdNumber"));
            System.out.println("City : "+rs.getString("City")+"\n");
            System.out.println("Room : "+rs.getString("Room"));
            System.out.println("Date of Check-In : "+rs.getObject("CheckIn"));
            System.out.println("Date of Check-Out : "+rs.getObject("CheckOut"));
            System.out.println();


        }


    
}
}

