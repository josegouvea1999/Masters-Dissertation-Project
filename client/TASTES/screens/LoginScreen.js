import { StyleSheet, Text, View, SafeAreaView, Pressable } from "react-native";
import React ,{useEffect} from "react";
import { LinearGradient } from "expo-linear-gradient";
import { Entypo } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import * as WebBrowser from "expo-web-browser";
import { makeRedirectUri } from 'expo-auth-session';
import axios from "axios";

const LoginScreen = () => {

  const navigation = useNavigation();

  useEffect(() => {

  },[])

  async function login ()  {
    try {
      const response = await axios.get("http://192.168.1.114:5000/login", {withCredentials: true});
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
        <View style={{ height: 80 }} />
        <Entypo
          style={{ textAlign: "center" }}
          name="spotify"
          size={80}
          color="white"
        />
        <Text
          style={{
            color: "white",
            fontSize: 40,
            fontWeight: "bold",
            textAlign: "center",
            marginTop: 40,
          }}
        >
          TASTES
        </Text>
        <Text
          style={{
            color: "white",
            fontSize: 20,
            textAlign: "center",
            marginTop: 10,
          }}
        >
          Discover your playlist trends
        </Text>
        <View style={{ height: 80 }} />
        <Pressable
        onPress={login}
          style={{
            backgroundColor: "#1DB954",
            padding: 10,
            marginLeft: "auto",
            marginRight: "auto",
            width: 300,
            borderRadius: 25,
            alignItems: "center",
            justifyContent: "center",
            marginVertical:10
          }}
        >
          <Text>Sign In with spotify</Text>
        </Pressable>

      </SafeAreaView>
    </LinearGradient>
  );
};

export default LoginScreen;

const styles = StyleSheet.create({});