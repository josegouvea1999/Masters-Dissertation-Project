import { Text, View, SafeAreaView, Pressable, Image } from "react-native";
import React from "react";
import { LinearGradient } from "expo-linear-gradient";
import { Entypo } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import * as WebBrowser from "expo-web-browser";
import { makeRedirectUri } from 'expo-auth-session';
import axios from "axios";

const SERVER_URL = "http://192.168.1.70:5000"

const LoginScreen = () => {

  const navigation = useNavigation();

  async function login ()  {
    try {
      const response = await axios.get(SERVER_URL + "/login", {withCredentials: true});
      const res = await WebBrowser.openAuthSessionAsync(response.data, redirectUrl = makeRedirectUri({
        scheme: 'myapp',
        path: 'redirect'
      }))

      if (res.type === 'success') {
        navigation.navigate('Main');
      }
    } catch (error) {
      console.log(error);
    } 
  }

  return (
    <LinearGradient colors={["#040306", "#131624"]} style={{ flex: 1 }}>
      <SafeAreaView>
        <View style={{ alignItems: 'center', marginTop: 100}} >
        <Image
        source={require('../assets/logo.png')}
        style={{ width: 300, height: 200 }} />
        <Text
          style={{
            color: "white",
            fontSize: 20,
            textAlign: "center",
          }}
        >
          Explore your playlist trends...
        </Text>
        <Pressable
        onPress={login}
          style={{
            backgroundColor: "#1DB954",
            padding: 12,
            marginLeft: "auto",
            marginRight: "auto",
            width: 300,
            borderRadius: 30,
            alignItems: "center",
            justifyContent: "center",
            marginVertical: 50
          }}
        >
          <Text>Sign In with Spotify</Text>
        </Pressable>

        <Entypo
          style={{ textAlign: "center", marginTop: 250 }}
          name="spotify"
          size={60}
          color="white"
        />
        </View>
      </SafeAreaView>
    </LinearGradient>
  );
};

export default LoginScreen;