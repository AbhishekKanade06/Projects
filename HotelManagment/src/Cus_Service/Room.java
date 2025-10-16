package Cus_Service;
import java.sql.*;
import java.util.Scanner;
import HOLTE.Hotel;

public class Room extends Hotel{
    Scanner sc=new Scanner(System.in);
    String room;
    String room_typ;
    int Room_no;
    public void get_room(Connection con,int customerId) throws SQLException{
        System.out.println("Choose you RoomType\n1.AC\n2.Non-AC\n");
        switch(sc.nextInt()){
            case 1:
                room_typ="Ac";
                break;
            case 2:
                room_typ="Non-Ac";
                break;
            default:
                break;        
        }
        String q4="select * from Room where r_status is null and Type=?";
        PreparedStatement pst=con.prepareStatement(q4);
        pst.setString(1, room_typ);
        ResultSet rs=pst.executeQuery();
        System.out.println("Please Choose  your Room ");
        System.out.println("Aviable Room : ");
        while(rs.next()){
            if(rs.wasNull()){
                System.out.println("Sorry no room are Aviable ");
            }
            else{
                System.out.println(rs.getInt("Room_no"));
            }
            
        }
        System.out.println("enter Room number : ");
        Room_no=sc.nextInt();
        String q5="update customer set status=?,Room_no=?,Room=? where customerId="+customerId;
        PreparedStatement pst1=con.prepareStatement(q5);
        pst1.setString(1, "CheckIn");
        pst1.setInt(2,Room_no);
        pst1.setString(3,room_typ);
        pst1.executeUpdate();
        String q6="update Room set r_status=?,customerId=? where Room_no="+Room_no;
        PreparedStatement ps0=con.prepareStatement(q6);
        ps0.setString(1, "Occupied");
        ps0.setInt(2,customerId);
        ps0.executeUpdate();
    }
    

}
