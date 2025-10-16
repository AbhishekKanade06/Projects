package HOLTE;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.Scanner;
import Cus_Service.Room;
import Cus_Service.Services;


 interface Functions{
   void getcust();
    void update();
}
class CreatId implements Functions{

    public final Object CheckIn = null;
    Scanner sc=new Scanner(System.in);
    String name;
    String ph_no;
    String address,Id_name,Id_number;
    int Cus_ID,customerId;
    LocalDate date=LocalDate.now();
    public void getcust() { 
        int ch;

        System.out.println("Enter your name ");
        name=sc.nextLine();
        System.out.println("Enter your phone number ");
        ph_no=sc.nextLine();
       
        System.out.println("chose your ID proof\n1.Aadhaar\n2.Driving License\n3.Voter ID ");
        ch=sc.nextInt();
         switch(ch){
            case 1:
                Id_name="Aadhaar";
                
                break;
            case 2:
                Id_name="Driving License";
    
                break;
            case 3:
                Id_name="Voter Id";
                break;        
        }
        System.out.println("enter your "+Id_name+" number : ");
        Id_number=sc.next();
        System.out.println("Enter your City ");
        address=sc.next();

        
    }
    public void update(){
        System.out.println("Enter your registrated name ");
        name=sc.nextLine();
        System.out.println("Enter your phone number ");
        ph_no=sc.nextLine();
        System.out.println("Enter your address ");
        address=sc.nextLine();
    }
   
    
    public static void getchoice(){
        System.out.println("1. For registration \n2. for show details\n3.for update for data \n4.CheckOut\n\tenter 0 exite \nEnter your choice : ");
    }
    public void show_details(Connection con) throws SQLException{
        String q="select * from customer where customerId=?";
        System.out.println("Enter your Customer Id  : ");
        Cus_ID=sc.nextInt();
        PreparedStatement ps=con.prepareStatement(q);
        ps.setInt(1, Cus_ID);
        ResultSet rs=ps.executeQuery();
        while(rs.next()){
            System.out.println();
            System.out.println("\nName : "+rs.getString("customerName"));
            System.out.println("ContactNo : "+rs.getString("contactNo"));
            System.out.println(rs.getString("IdName")+"No. :"+rs.getString("IdNumber"));
            System.out.println("City : "+rs.getString("City")+"\n");
            System.out.println("Room : "+rs.getString("Room"));
            System.out.println("Date of Check-In : "+rs.getObject("CheckIn"));
            System.out.println();


        }   
    }
    public void checkout(Connection con) throws SQLException{
        Services s=new Services();
        System.out.print("Enter you customer Id : ");
        int cus=sc.nextInt();
        LocalDate date=LocalDate.now();
        String q4="update customer set status=?,CheckOut=? where customerId="+cus;
        PreparedStatement ps=con.prepareStatement(q4);
        ps.setString(1, "CheckOut");
        ps.setObject(2,date);
        ps.executeUpdate();
        String q6="update Room set r_status=null,customerId=null where customerId="+cus;
        PreparedStatement ps0=con.prepareStatement(q6);
        ps0.executeUpdate();
        String q7="update bill_cus_"+cus+" set Date_of_end=?";
        PreparedStatement ps7=con.prepareStatement(q7);
        ps7.setObject(1,date);
        ps7.executeUpdate();
        s.get_bill(con,cus);
        System.out.println("\n\n\t\tPlease visit Again \n\n");

    }

    
}

public class Hotel {
    public static void main(String[] args) throws SQLException, ClassNotFoundException{
        Scanner sc=new Scanner(System.in);
        Class.forName("com.mysql.cj.jdbc.Driver");
        String url="jdbc:mysql://localhost:3306/hotle";
        String user="root";
        String  pass="a.k.47576";
        Connection con = DriverManager.getConnection(url, user, pass);
        System.out.println("\n\n!!!!Welcome!!\n");
        int n;
        CreatId.getchoice();
        CreatId c0=new CreatId();
        n=sc.nextInt();
        while(n!=0){
            
            switch(n){
                case 1:
                    
                    c0.getcust();

                    String q="insert into customer(customerName,contactNo,IdName,IdNumber,City,CheckIn) values(?,?,?,?,?,?)";
                    PreparedStatement stmst=con.prepareStatement(q);

                    stmst.setString(1, c0.name);
                    stmst.setString(2,c0.ph_no);
                    
                    stmst.setString(3,c0.Id_name);
                    stmst.setString(4,c0.Id_number);
                    stmst.setString(5,c0.address);
                    stmst.setObject(6,c0.date);

                    stmst.executeUpdate();
                    
                    String q0="select customerId from customer where IdNumber=?";
                    PreparedStatement ps=con.prepareStatement(q0);
                    ps.setString(1, c0.Id_number);
                    ResultSet r=ps.executeQuery();
                    while(r.next()){
                        c0.customerId=r.getInt("customerId");
                        System.out.println("Your Customer Id is : "+r.getInt("customerId"));
                    }
                    System.out.println("Thanyou\nYour regestration is compeleted\n\n ");
                    Room r1=new Room();
                    r1.get_room(con,c0.customerId);
                    Services s=new Services();
                    s.create_bill(con, c0.customerId);
                    s.get_services(con,c0.customerId);
                    break;
                case 3:
                    CreatId c1=new CreatId();
                    c1.update();
                    String q1="update customer set contactNo=?,City=? where customerName=?";
                    PreparedStatement st1=con.prepareStatement(q1);
                    st1.setString(3,c0.name);
                    st1.setString(1, c0.ph_no);
                    st1.setString(2, c0.address);
                    st1.executeUpdate();
                    System.out.println("\nYour data get updated \n");
                    break;
                case 2:
                     
                        c0.show_details(con);
                        break;
                case 4:
                    c0.checkout(con);
        }
        if(n==4){
            n=0;
        }
        else{
            System.out.println();
            CreatId.getchoice();
            n=sc.nextInt();
        }
    }
    con.close();
    sc.close();
}
}

