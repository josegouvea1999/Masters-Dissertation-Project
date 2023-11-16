import { StyleSheet, Text, ScrollView, View, Image, Pressable } from "react-native";
import React ,{useEffect, useState} from "react";
import { LinearGradient } from "expo-linear-gradient";
import { useNavigation } from "@react-navigation/native";
import { AntDesign } from "@expo/vector-icons";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import axios from "axios";


const HomeScreen = () => {

  const [playlists, setPlaylists] = useState([{}]);
  const [playlist_covers, setPlaylistsCovers] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const navigation = useNavigation();

  useEffect(() => {
    getProfile();
    getPlaylists();
  },[])

  const greetingMessage = () => {
    const currentTime = new Date().getHours();
    if (currentTime < 12) {
      return "Good Morning";
    } else if (currentTime < 16) {
      return "Good Afternoon";
    } else {
      return "Good Evening";
    }
  };
  const message = greetingMessage();

  const getProfile = async () => {
    try {
      const response = await axios.get("http://192.168.1.114:5000/profile");

      const data = await response.data;

      setUserProfile(data);

      return data;

    } catch (error) {
      console.log(error);
    }

  }

  const getPlaylists = async () => {
    try {
      const response = await axios.get("http://192.168.1.114:5000/playlists");

      const data = await response.data;

      await setPlaylists(data.items);

      const playlists_uri = [];

      for (let i = 0; i < playlists.length; i++) {
        playlists_uri.push(data.items[i].images[0].url);
      }

      setPlaylistsCovers(playlists_uri)
      
      console.log(playlists);

    } catch (error) {
      console.log(error);
    } 
  
  }

  const refresh = async (id) => {
    console.log(id);
  }


  return (
    <LinearGradient colors={["#040306", "#131624"]} style={{ flex: 1 }}>
        <View
          style={{
            marginTop: 30,
            padding: 10,
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Image
              style={{
                width: 40,
                height: 40,
                borderRadius: 20,
                resizeMode: "cover",
              }}
              source={{ uri: userProfile?.images[0].url }}
            />
            <Text
              style={{
                marginLeft: 20,
                fontSize: 20,
                fontWeight: "bold",
                color: "white",
              }}
            >
              {message +  " " + userProfile?.display_name.split(" ", 2)[0] + " !"}
            </Text>
          </View>

          <MaterialCommunityIcons
            name="lightning-bolt-outline"
            size={24}
            color="white"
          />
        </View>

        <ScrollView style={{ marginTop: 20 }}>


        {playlists.map((playlist, i) => (
        <Pressable
          key={i}
          onPress={() => refresh(playlist.id)}
          style={{
            marginBottom: 10,
            flexDirection: "row",
            alignItems: "center",
            gap: 10,
            flex: 1,
            marginHorizontal: 10,
            marginVertical: 3,
            backgroundColor: "#202020",
            borderRadius: 4,
            elevation: 3,
          }}
        >
          <LinearGradient colors={["#33006F", "#FFFFFF"]}>
            <Pressable
              style={{
                width: 75,
                height: 75,
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {/* Assuming 'images' is an array and we're using the first image for the playlist */}
              <Image
                source={{ uri: playlist_covers[playlists.indexOf(playlist)]  }} // Use the URL from the playlist's images array
                style={{ width: 75, height: 75 }} // Adjust the size as per your requirement
              />
            </Pressable>
          </LinearGradient>

          <Text style={{ color: "white", fontSize: 15, fontWeight: "bold", maxWidth: '80%' }}>
            {playlist.name}
          </Text>
        </Pressable>
      ))}
      </ScrollView>
    </LinearGradient>
  );
};

export default HomeScreen;

const styles = StyleSheet.create({});